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
from att2seq_predictor import Att2SeqPredictor
from utils import DataLoader
from transformers import GPT2Tokenizer



def main(args):
    print('loading...')
    predictor = Att2SeqPredictor(
        args.model_weight_dir,
        args.data_path,
        args.index_dir,
        20000
    )
    corpus = DataLoader(
        args.data_path, 
        args.index_dir, 
        20000
        )
    aspect_templates = joblib.load(args.aspect_template_dir)
    aspect_templates = np.array(aspect_templates)

    print('ranking...')
    rankings = []

    for sample in tqdm(corpus.test[:args.max_ranking_sample]):
        user, item, rating, seq, feature = sample['user'], sample['item'], sample['rating'], \
            predictor.seq2language(sample['text']), sample['feature']
        
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

    mrr = np.mean(1/(1+np.array(rankings)) )
    print('mrr: ',mrr)
    return mrr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None,
                        help='path for loading the pickle data')
    parser.add_argument('--index_dir', type=str, default=None,
                        help='load indexes')
    parser.add_argument('--aspect_template_dir', type=str, default=None,
                        help='dir to templates with emptied aspect')
    parser.add_argument('--model_weight_dir', type=str, default=None,
                        help='model weight dir')
    parser.add_argument('--max_ranking_sample', type=int, default=100,
                        help='maximum test sample to run ranking on')
    parser.add_argument('--output_dir', type=str, default='./tmp',
                        help='directory to put outputs to')
    args = parser.parse_args()
    mrr = main(args)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'ranking_result.txt'), 'w') as ofp:
        ofp.write('MRR: '+str(mrr)+'\n')
        ofp.write(str(args))
    