"""
Interface for using trained att2seq
"""
import os
import math
import torch
import argparse
import joblib
import torch.nn as nn
import numpy as np
import subprocess
from tqdm import tqdm
from peter_predictor import PeterPredictor
from utils import DataLoader
from transformers import GPT2Tokenizer
import pandas as pd
from ranking import main as ranking_main
from entailment import main as entailment_main
from rate_negations import main as rate_negations_main

def load_predictor(args):
    print('loading model...')
    predictor = PeterPredictor(
        args.model_weight_dir,
        args.data_path,
        args.index_dir,
        20000
    )
    return predictor 

def load_corpus(args):
    print('loading corpus...')
    corpus = DataLoader(
        args.data_path, 
        args.index_dir, 
        20000
        )
    return corpus 

def generate(args):
    print('generating text')

    predictor = load_predictor(args)
    corpus = load_corpus(args)

    instances = []
    for sample in tqdm(corpus.test[:args.max_test_sample]):
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

        instances.append(this_instance)
    
    out = pd.DataFrame(instances)

    return out

def main(args):
    # handle path
    if args.output_dir == 'auto':
        args.output_dir = os.path.join(os.path.split(args.model_weight_dir)[-2],  'evaluations')
        print('redirecting output to', args.output_dir)

    # # generate text
    # generated = generate(args)
    # os.makedirs(args.output_dir, exist_ok = True)
    # generated.to_csv(os.path.join(args.output_dir, 'generated.csv'))

    # mrr ranking
    args.aspect_template_dir = os.path.join(args.index_dir, 'test_aspect_templates.joblib')
    mrr = ranking_main(args)
    with open(os.path.join(args.output_dir, 'ranking_result.txt'), 'w') as ofp:
        ofp.write('MRR: '+str(mrr)+'\n')
        ofp.write(str(args))

    # # entailment
    # entailment_main(args)

    # # negations
    # rate_negations_main(args)

    # # semantic evaluations, needs bartscore environment
    # import sys
    # sys.path.insert(0,'BARTScore')
    # from score_from_files import main as semantic_score_main
    # from easydict import EasyDict as edict
    # semantic_score_main(
    #     edict({
    #         'test_csv_dir':os.path.join(args.output_dir, 'generated.csv'), 
    #         'score_with':'bert',
    #         'referencefield_key':'review',
    #         'textfield_key':'text_predicted'
    #         })
    # )
    # semantic_score_main(
    #     edict({
    #         'test_csv_dir':os.path.join(args.output_dir, 'generated.csv'), 
    #         'score_with':'bart',
    #         'referencefield_key':'review',
    #         'textfield_key':'text_predicted'
    #         })
    # )
    # semantic_score_main(
    #     edict({
    #         'test_csv_dir':os.path.join(args.output_dir, 'generated.csv'), 
    #         'score_with':'mauve',
    #         'referencefield_key':'review',
    #         'textfield_key':'text_predicted'
    #         })
    # )

    # # text-label agreement
    # import sys
    # sys.path.insert(0,'bert_regression')
    # from predict_with_gold import main as text_label_agreement_main
    # from easydict import EasyDict as edict
    # import numpy as np
    # tl_agreement_mse = text_label_agreement_main(
    #     edict({
    #         'model':'bert',
    #         'test_csv_dir':os.path.join(args.output_dir, 'generated.csv'), 
    #         'textfield_key':'text_predicted',
    #         'labelfield_key':'rating',
    #         'weight_dir':os.path.join(args.regression_model_dir, 'weights.pt'),
    #         'regression':True,
    #         'batch_size':32,
    #         'max_seq_len':100,
    #         'output_dir':args.output_dir
    #         })
    # )
    # with open(os.path.join(args.output_dir, 'tl_agreement.txt'), 'w') as ofp:
    #     ofp.write('mse: '+str(tl_agreement_mse)+'\n')
    #     ofp.write('evaluated with :'+args.regression_model_dir)

    # # hallucination
    # from hallucination_nli import main as hallucination_nli_main
    # print('evaluating factual hallucination...')
    # hallucination_nli_main(args)
    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Att2Seq (EACL\'17) without rating input')
    parser.add_argument('--data_path', type=str, default=None,
                        help='path for loading the pickle data')
    parser.add_argument('--index_dir', type=str, default=None,
                        help='load indexes')
    parser.add_argument('--model_weight_dir', type=str, default=None,
                        help='model weight dir')
    parser.add_argument('--max_test_sample', type=int, default=100,
                        help='maximum test sample to run on')
    parser.add_argument('--max_ranking_sample', type=int, default=100,
                        help='maximum test sample to run ranking on')
    parser.add_argument('--output_dir', type=str, default='./tmp',
                        help='directory to put outputs to')
    parser.add_argument('--regression_model_dir', type=str, default='./tmp',
                        help='directory of regression model to evaluate text-label agreement')
    args = parser.parse_args()
    main(args)





