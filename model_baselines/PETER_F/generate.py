import os
import math
import torch
import argparse
import torch.nn as nn
import numpy as np
import pandas as pd
from module import PETER
from tqdm import tqdm
from utils import  DataLoader, Batchify,  ids2tokens
from peter_predictor import PeterPredictor, generate


def main():
    
    print('loading')
    predictor = PeterPredictor(
        '/home/zhouhang/data/review-generation/PETER/model/tripadvisorf_backup/model.pt',
        './data/TripAdvisor/reviews.pickle',
        './data/TripAdvisor/1/',
        20000
    )
    corpus = DataLoader(
        './data/TripAdvisor/reviews.pickle', 
        './data/TripAdvisor/1/', 
        20000
        )

    print('predicting')

    instances = []
    for sample in tqdm(corpus.test[:10000]):
        user, item, rating, seq, feature = sample['user'], sample['item'], sample['rating'], \
            predictor.seq2language(sample['text']), predictor.idx2word[sample['feature']] 
        
        this_instance = {
                'userid':user,
                'itemid':item,
                'user':user,
                'item':item,
                'rating':rating,
                'review':seq,
                'feature':feature
            }
        user, item, rating, seq, feature = predictor.tensorize(this_instance)
        text_p, rating_p = predictor.generate(user, item, rating, seq, feature)
        
        this_instance['text_predicted'] = text_p 
        this_instance['rating_predicted'] = rating_p

        instances.append(this_instance)
    
    out = pd.DataFrame(instances)
    out.to_csv('generated.csv')

if __name__ == '__main__':
    main()