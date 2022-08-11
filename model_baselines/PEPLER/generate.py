import os
import math
import torch
import argparse
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import  DataLoader, Batchify,  ids2tokens
from pepler_predictor import PeplermfPredictor
from transformers import GPT2Tokenizer


def main():
    
    print('loading')
    predictor = PeplermfPredictor(
            model_path = '/home/zhouhang/data/review-generation/PEPLER/tripadvisormf/model.pt'
        )
    bos = '<bos>'
    eos = '<eos>'
    pad = '<pad>'
    corpus = DataLoader(
            '../nlg4rec_data/TripAdvisor/reviews.pickle', 
            '../nlg4rec_data/TripAdvisor/1/', 
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token=bos, eos_token=eos, pad_token=pad),
            seq_len=20
        )

    print('predicting')

    instances = []
    for sample in tqdm(corpus.test[:10000]):
        user, item, rating, seq = sample['user'], sample['item'], sample['rating'], sample['text']
            
        
        this_instance = {
                'userid':user,
                'itemid':item,
                'user':user,
                'item':item,
                'rating':rating,
                'review':seq
            }
        
        generated = predictor.generate(user, item)
        this_instance['text_predicted'] = generated['text_predicted']
        this_instance['rating_predicted'] = generated['rating_predicted']

        instances.append(this_instance)
        
    
    out = pd.DataFrame(instances)
    out.to_csv('./tripadvisormf/generated.csv')

if __name__ == '__main__':
    main()