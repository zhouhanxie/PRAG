"""
Module to let PETER rate nagations generated by
https://huggingface.co/dapang/yelp_pos2neg_lm_bart_large?text=the+hotel+may+be+under+renovation+but+the+suite+is+gorgeous
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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
        20000
    )
    corpus = DataLoader(
        args.data_path, 
        args.index_dir, 
        20000
        )

    # negation model
    model_name = 'dapang/yelp_pos2neg_lm_bart_large'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    def generate_batch_negation(examples):
        batch = tokenizer(examples, max_length=1024, padding=True, truncation=True, return_tensors="pt")
        out = model.generate(batch['input_ids'].to(model.device), num_beams=5)
        negative_examples = tokenizer.batch_decode(out, skip_special_tokens=True)
        return negative_examples

    print('rating...')
    rankings = []

    flip_counter = 0
    n_samples = args.max_test_sample
    out_string = ""
    positive_samples = [i for i in corpus.test if i['rating'] >= 4.0 and i['rating'] <=5.0 ]
    for i,sample in tqdm(enumerate(positive_samples[:n_samples]), total=n_samples):
        out_string = out_string+'\n## '+str(i+1)+'\\'+str(n_samples)+' ##\n'
        user, item, rating, seq, feature = sample['user'], sample['item'], sample['rating'], \
                predictor.seq2language(sample['text']), predictor.idx2word[sample['feature']] 
        ppl = predictor.ppl_from_input(dictorize(user, item, rating, seq, feature))
        seq_hat = generate_batch_negation([seq])[0]
        ppl_hat = predictor.ppl_from_input(dictorize(user, item, rating, seq_hat, feature))
        out_string += 'original: '+ seq + '\n'
        out_string += str(ppl)+'\n'
        out_string += 'negation: '+ seq_hat + '\n'
        out_string += str(ppl_hat)+'\n'
        if ppl_hat < ppl:
            flip_counter += 1
    out_string = out_string +'done, inconsistent labels: '+str(flip_counter)+'\n'
    

    with open(os.path.join(args.output_dir, 'rate_negations.out'),'w') as ofp:
        ofp.write(out_string)

    print('done, inconsistent labels: '+str(flip_counter))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main()
    