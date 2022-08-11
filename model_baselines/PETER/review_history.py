import os
import joblib
import torch
import pickle
import numpy as np
np.random.seed(0)
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from utils import *

class DataReader:
    def __init__(self, data_path, index_dir):
        self.user_dict = EntityDictionary()
        self.item_dict = EntityDictionary()
        self.max_rating = float('-inf')
        self.min_rating = float('inf')
        self.initialize(data_path)
        self.feature_set = set()
        self.train, self.valid, self.test, self.user2feature, self.item2feature = self.load_data(data_path, index_dir)

    def initialize(self, data_path):
        assert os.path.exists(data_path)
        reviews = pickle.load(open(data_path, 'rb'))
        for review in reviews:
            self.user_dict.add_entity(review['user'])
            self.item_dict.add_entity(review['item'])
            rating = review['rating']
            if self.max_rating < rating:
                self.max_rating = rating
            if self.min_rating > rating:
                self.min_rating = rating

    def load_data(self, data_path, index_dir):
        data = []
        reviews = pickle.load(open(data_path, 'rb'))
        for review in reviews:
            (fea, adj, text, sco) = review['template']
            data.append({'user': self.user_dict.entity2idx[review['user']],
                         'item': self.item_dict.entity2idx[review['item']],
                         'rating': review['rating'],
                         'text': text,
                         'feature': fea})
            self.feature_set.add(fea)

        train_index, valid_index, test_index = self.load_index(index_dir)
        train, valid, test = [], [], []
        user2feature, item2feature = {}, {}
        for idx in train_index:
            review = data[idx]
            train.append(review)
            u = review['user']
            i = review['item']
            f = review['feature']
            if u in user2feature:
                user2feature[u].append(f)
            else:
                user2feature[u] = [f]
            if i in item2feature:
                item2feature[i].append(f)
            else:
                item2feature[i] = [f]
        for idx in valid_index:
            valid.append(data[idx])
        for idx in test_index:
            test.append(data[idx])
        return train, valid, test, user2feature, item2feature

    def load_index(self, index_dir):
        assert os.path.exists(index_dir)
        with open(os.path.join(index_dir, 'train.index'), 'r') as f:
            train_index = [int(x) for x in f.readline().split(' ')]
        with open(os.path.join(index_dir, 'validation.index'), 'r') as f:
            valid_index = [int(x) for x in f.readline().split(' ')]
        with open(os.path.join(index_dir, 'test.index'), 'r') as f:
            test_index = [int(x) for x in f.readline().split(' ')]
        return train_index, valid_index, test_index

class ReviewHistory:
    """
    a class for fetching review history
    at the cost of ram
    """
    def __init__(self, data, valid_data=None, test_data=None):
        self.text_table = []
        self.user2_texttableids = defaultdict(list)
        self.item2_texttableids = defaultdict(list)
        self.ui2_texttableid = dict()
        self.user2item = defaultdict(list)
        self.item2user = defaultdict(list)
        
        cur_texttable_id = 0
        self.text_table.append('') # stub empty string
        cur_texttable_id += 1
        ratings = []
        for sample in data:
            self.text_table.append(sample['text'])
            self.user2_texttableids[sample['user']].append(cur_texttable_id)
            self.item2_texttableids[sample['item']].append(cur_texttable_id)
            self.ui2_texttableid[(sample['user'], sample['item'])] = cur_texttable_id
            self.user2item[sample['user']].append(sample['item'])
            self.item2user[sample['item']].append(sample['user'])
            cur_texttable_id += 1
            ratings.append(sample['rating'])
        self._mean_rating = np.average(ratings)

        # if valid and test data is not None, extend text table to
        # keep those reviews as well.
        # but do not keep those in history.
        if valid_data is not None:
            for sample in valid_data:
                self.text_table.append(sample['text'])
                self.ui2_texttableid[(sample['user'], sample['item'])] = cur_texttable_id
                cur_texttable_id += 1

        if test_data is not None:
            for sample in test_data:
                self.text_table.append(sample['text'])
                self.ui2_texttableid[(sample['user'], sample['item'])] = cur_texttable_id
                cur_texttable_id += 1
            
        self.text_table = np.array(self.text_table, dtype=object)

    def nuser(self):
        return len(self.user2item)

    def nitem(self):
        return len(self.item2user)

    def nreview(self):
        return len(self.text_table)

    def mean_rating(self):
        return self._mean_rating

    def load_embedded_text_table(self, table_dir):
        self.raw_text_table = np.array(
            [t for t in self.text_table]
        )
        with open(table_dir, 'rb') as f:
            self.text_table = np.load(f)
        
        
    def build_embedded_text_table(self, sentence_transformer, device, maybe_load_from=None):
        if maybe_load_from is not None:
            print('trying to load reviews')
            try:
                self.load_embedded_text_table(maybe_load_from)
                return self
            except Exception as e:
                print(e)
                print('loading failed')

        print('embedding reviews')
        self.raw_text_table = np.array(
            [t for t in self.text_table]
        )
        self.text_table = sentence_transformer.encode(
            self.text_table.tolist(), 
            convert_to_numpy=True, 
            device=device, 
            show_progress_bar=True
        )
        if maybe_load_from is not None:
            np.save(open(maybe_load_from, 'wb'), self.text_table)
        return self

    def load_embedded_text_table(self, table_path):
        print('loading embedded reviews')
        self.raw_text_table = np.array(
            [t for t in self.text_table]
        )
        self.text_table = np.load(table_path)
        return self
        
            
    def get_user_history(self, user, hide_item=None, truncation=True, return_embedding=True):
        
        if hide_item != None and (user, hide_item) in self.ui2_texttableid:
            history = [i for i in self.user2_texttableids[user] if i != self.ui2_texttableid[(user, hide_item)]]
        else:
            history = [i for i in self.user2_texttableids[user]]
         
        if not return_embedding:
            return history
        if truncation:
            return np.mean(self.text_table[history], axis=-1)
        return self.text_table[history]
    
    def get_item_history(self, item, hide_user=None, truncation=True, return_embedding=True):
        
        if hide_user != None and (hide_user, item) in self.ui2_texttableid:
            history = [i for i in self.item2_texttableids[item] if i != self.ui2_texttableid[(hide_user, item)]]
        else:
            history = [i for i in self.item2_texttableids[item]]
        
        if not return_embedding:
            return history
        if truncation:
            return np.mean(self.text_table[history], axis=-1)
        return self.text_table[history]
    
    def get_ui_review(self, user, item, return_embedding=True):
        if return_embedding:
            return self.text_table[self.ui2_texttableid[(user, item)]]
        return self.ui2_texttableid[(user, item)]