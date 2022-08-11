import os
import math
import torch
import argparse
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import  *
from nrt_predictor import NRTPredictor


def main():
    
    print('loading')
    predictor = NRTPredictor(
        '/home/zhouhang/data/review-generation/NRT/tripadvisor/model.pt',
        '../nlg4rec_data/TripAdvisor/reviews.pickle',
        '../nlg4rec_data/TripAdvisor/1/',
        20000
    )
    corpus = DataLoader(
        '../nlg4rec_data/TripAdvisor/reviews.pickle', 
        '../nlg4rec_data/TripAdvisor/1/', 
        20000
        )

    print('predicting')

    instances = []
    for sample in tqdm(corpus.test[:10000]):
        user, item, rating, seq, feature = sample['user'], sample['item'], sample['rating'], \
            predictor.seq2language(sample['text']), sample['feature'] 
        
        this_instance = {
                'userid':user,
                'itemid':item,
                'user':user,
                'item':item,
                'rating':rating,
                'review':seq,
                'feature':feature
            }
        user, item, seq = predictor.tensorize(this_instance)
        generated = predictor.generate(user, item)
        
        this_instance['text_predicted'] = generated['text_predicted']
        this_instance['rating_predicted'] = generated['rating_predicted']

        instances.append(this_instance)
    
    out = pd.DataFrame(instances)
    out.to_csv('./tripadvisor/generated.csv')

if __name__ == '__main__':
    main()