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



def has_tensor(obj) -> bool:
    """
    Given a possibly complex data structure,
    check if it has any torch.Tensors in it.
    Credit: AllenNLP
    """
    if isinstance(obj, torch.Tensor):
        return True
    elif isinstance(obj, dict):
        return any(has_tensor(value) for value in obj.values())
    elif isinstance(obj, (list, tuple)):
        return any(has_tensor(item) for item in obj)
    else:
        return False

def move_to_device(obj, cuda_device):
    """
    Given a structure (possibly) containing Tensors on the CPU,
    move all the Tensors to the specified GPU (or do nothing, if they should be on the CPU).
    Credit: AllenNLP
    """

    if cuda_device == torch.device("cpu") or not has_tensor(obj):
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.to(cuda_device)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, cuda_device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, cuda_device) for item in obj]
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # This is the best way to detect a NamedTuple, it turns out.
        return obj.__class__(*(move_to_device(item, cuda_device) for item in obj))
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, cuda_device) for item in obj)
    else:
        return obj


def pad_sequence_to_length(
    sequence,
    desired_length,
    default_value = lambda: 0,
    padding_on_right = True,
) :
    """
    Take a list of objects and pads it to the desired length, returning the padded list.  The
    original list is not modified.
    # Parameters
    sequence : `List`
        A list of objects to be padded.
    desired_length : `int`
        Maximum length of each sequence. Longer sequences are truncated to this length, and
        shorter ones are padded to it.
    default_value: `Callable`, optional (default=`lambda: 0`)
        Callable that outputs a default value (of any type) to use as padding values.  This is
        a lambda to avoid using the same object when the default value is more complex, like a
        list.
    padding_on_right : `bool`, optional (default=`True`)
        When we add padding tokens (or truncate the sequence), should we do it on the right or
        the left?
    # Returns
    padded_sequence : `List`
    credit: AllenNLP
    """
    sequence = list(sequence)
    # Truncates the sequence to the desired length.
    if padding_on_right:
        padded_sequence = sequence[:desired_length]
    else:
        padded_sequence = sequence[-desired_length:]
    # Continues to pad with default_value() until we reach the desired length.
    pad_length = desired_length - len(padded_sequence)
    # This just creates the default value once, so if it's a list, and if it gets mutated
    # later, it could cause subtle bugs. But the risk there is low, and this is much faster.
    values_to_pad = [default_value()] * pad_length
    if padding_on_right:
        padded_sequence = padded_sequence + values_to_pad
    else:
        padded_sequence = values_to_pad + padded_sequence
    return padded_sequence

class EntityDictionary:
    def __init__(self):
        self.idx2entity = []
        self.entity2idx = {}

    def add_entity(self, e):
        if e not in self.entity2idx:
            self.entity2idx[e] = len(self.idx2entity)
            self.idx2entity.append(e)

    def __len__(self):
        return len(self.idx2entity)

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
        self.feature_table = []
        self.user2_texttableids = defaultdict(list)
        self.item2_texttableids = defaultdict(list)
        self.ui2_texttableid = dict()
        self.user2item = defaultdict(list)
        self.item2user = defaultdict(list)
        
        cur_texttable_id = 0
        self.text_table.append('') # stub empty string
        cur_texttable_id += 1
        ratings = []
        cur_featuretable_id = 0
        self.feature_table.append('') #offset
        cur_featuretable_id += 1
        
        for sample in data:
            self.text_table.append(sample['text'])
            self.feature_table.append(sample['feature'])
            self.user2_texttableids[sample['user']].append(cur_texttable_id)
            self.item2_texttableids[sample['item']].append(cur_texttable_id)
            self.ui2_texttableid[(sample['user'], sample['item'])] = cur_texttable_id
            self.user2item[sample['user']].append(sample['item'])
            self.item2user[sample['item']].append(sample['user'])
            cur_texttable_id += 1
            cur_featuretable_id += 1
            ratings.append(sample['rating'])
        self._mean_rating = np.average(ratings)
        ratings.insert(0, self._mean_rating) # offset for the stub review
        self.rating_table = np.array(ratings)

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
        self.feature_table = np.array(self.feature_table, dtype=object)

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
    
    

def build_dataset(data, history, tokenizer):
        
        datapoints = []
        for sample in tqdm(data):
            
            # get all prev reviews for that item, except one written by current user
            review_history = history.get_item_history(sample['item'], hide_user=sample['user'], return_embedding=False)

            # the gold is user's review for that item, negative is user's review for other things
            negatives = history.get_user_history(sample['user'], hide_item=sample['item'], return_embedding=False)
            negatives = np.random.choice(negatives, size=1)
            positive = history.get_ui_review(sample['user'], sample['item'], return_embedding=False)
            
            
            datapoints.append({
                    'review_id':positive,
                    'review_history_ids':review_history,
                    'is_positive':1
                })
            for user_review in negatives:
                datapoints.append({
                    'review_id':user_review,
                    'review_history_ids':review_history,
                    'is_positive':0
                })
    
        
        
        longest_history = max([len(dp['review_history_ids']) for dp in datapoints])
        padded_history = [pad_sequence_to_length(dp['review_history_ids'], longest_history) for dp in datapoints]
        
        tensor_dataset = TensorDataset(
            torch.tensor([dp['review_id'] for dp in datapoints]),
            torch.tensor(padded_history),
            torch.tensor([dp['is_positive'] for dp in datapoints])
        )

        
        return tensor_dataset


def get_dataloaders(
    data_path, 
    index_dir, 
    batch_size=128
):

    data = DataReader(
        data_path = data_path,
        index_dir = index_dir
    )
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    
    review_history = review_history = ReviewHistory(data.train, valid_data=data.valid, test_data=data.test)
    
    # note: the ".train" here is train data of public splits
    # we well further split this for training retrievers
    dataset = build_dataset(data.train, review_history, tokenizer)  

    val_dataset = build_dataset(data.valid, review_history, tokenizer)  

    test_dataset = build_dataset(data.test, review_history, tokenizer)  
    
    # # create splitted dataloaders
    # train_size = int(0.8 * len(dataset))
    # val_and_test_size = len(dataset) - train_size
    # val_size = int(0.75 * val_and_test_size)
    # test_size = val_and_test_size - val_size
    
    # print('train/val/test split num: ', train_size, val_size, test_size)
    
    # train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
        
    
        
    return {
        'train_dataloader':train_dataloader,
        'val_dataloader':val_dataloader,
        'test_dataloader':test_dataloader,
        'review_history':review_history
    }