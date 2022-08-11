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
from module import PETER
from tqdm import tqdm
from peter_predictor import PeterPredictor
from utils import DataLoader

def dictorize(user, item, rating, seq, feature):
    out = {
        'userid':user,
        'itemid':item,
        'rating':rating,
        'feature':feature,
        'review':seq
    }
    return out

def main(args):
    print('loading...')
    predictor = PeterPredictor(
        args.model_weight_dir,
        args.data_path,
        args.index_dir,
        20000,
        use_feature=False
    )
    aspect_templates = joblib.load(args.aspect_template_dir)
    corpus = DataLoader(
        args.data_path, 
        args.index_dir, 
        20000
        )
    aspect_templates = np.array(aspect_templates)

    print('ranking...')
    rankings = []

    for sample in tqdm(corpus.test[:args.max_ranking_sample]):
        try:
            user, item, rating, seq, feature = sample['user'], sample['item'], sample['rating'], \
                predictor.seq2language(sample['text']), predictor.idx2word[sample['feature']] 
            
            adv_template_idx = np.random.randint(low=0, high=len(aspect_templates)-1, size=100, dtype=int)
            adv_seqs = [s.replace('@ASPECT', feature) for s in aspect_templates[adv_template_idx]]
            all_seqs = [seq] + adv_seqs
            all_ppls = []
            for x in all_seqs:
                ppl_hat = predictor.ppl_from_input(dictorize(user, item, rating, x, feature))
                all_ppls.append(ppl_hat)
            ranking = np.argwhere(
                np.array(all_seqs)[np.argsort(all_ppls)] == seq
            )[0].item()
            rankings.append(ranking)
        except KeyboardInterrupt:
            raise 
        except:
            print('WARNING: opps, an err occured')

    mrr = np.mean(1/(1+np.array(rankings)) )
    print('mrr: ',mrr)
    return mrr

if __name__ == '__main__':
    main()
    # should give mrr:  0.20253455011427518