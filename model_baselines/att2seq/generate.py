import os
import math
import torch
import argparse
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import  DataLoader, Batchify,  ids2tokens
from att2seq_predictor import Att2SeqPredictor


def main():
    
    print('loading')
    predictor = Att2SeqPredictor(
        '/home/zhouhang/data/review-generation/Att2Seq/tripadvisor/model.pt',
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
        user, item, seq, rating = sample['user'], sample['item'], predictor.seq2language(sample['text']), sample['rating']
        
        this_instance = {
                'userid':user,
                'itemid':item,
                'user':user,
                'item':item,
                'rating':rating,
                'review':seq,    
            }

        text_p = predictor.generate(user, item)
        
        this_instance['text_predicted'] = text_p 

        instances.append(this_instance)
    
    out = pd.DataFrame(instances)
    out.to_csv('generated.csv')

if __name__ == '__main__':
    main()