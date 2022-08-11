"""
Groundtruth ranking with adversarial data
"""
import os
import math
import torch
import argparse
import joblib
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pepler_predictor import PeplermfPredictor
from utils import DataLoader
from transformers import GPT2Tokenizer



def main(args):
    print('loading...')
    predictor = PeplermfPredictor(
        model_path = args.model_weight_dir
    )
    bos = '<bos>'
    eos = '<eos>'
    pad = '<pad>'
    corpus = DataLoader(
            args.data_path, 
            args.index_dir, 
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token=bos, eos_token=eos, pad_token=pad),
            seq_len=20
        )
    aspect_templates = joblib.load(args.aspect_template_dir)
    aspect_templates = np.array(aspect_templates)

    print('ranking...')
    rankings = []

    for sample in tqdm(corpus.test[:args.max_ranking_sample]):
        try:
            user, item, rating, seq, feature = sample['user'], sample['item'], sample['rating'], \
                sample['text'], sample['feature'] 
            
            adv_template_idx = np.random.randint(low=0, high=len(aspect_templates)-1, size=100, dtype=int)
            adv_seqs = [s.replace('@ASPECT', feature) for s in aspect_templates[adv_template_idx]]
            all_seqs = [seq] + adv_seqs
            all_ppls = []
            for x in all_seqs:
                ppl_hat = predictor.ppl(user, item, x)
                all_ppls.append(ppl_hat)
            ranking = np.argwhere(
                np.array(all_seqs)[np.argsort(all_ppls)] == seq
            )[0].item()
            rankings.append(ranking)
        except KeyboardInterrupt:
            raise 
        except:
            print('WARNING: some unknown error occured')

    mrr = np.mean(1/(1+np.array(rankings)) )
    print('mrr: ',mrr)
    return mrr

if __name__ == '__main__':
    main()

    # mrr: 0.16264287628022486