"""
obtain some test metrics
that are already in the repo
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
from att2seq_predictor import Att2SeqPredictor
from utils import DataLoader
from transformers import GPT2Tokenizer
import pandas as pd
from utils import rouge_score, bleu_score, DataLoader, Batchify, now_time, ids2tokens, unique_sentence_percent, \
    feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity
device = torch.device('cuda') 

def evaluate(model, data):
    model.eval()
    text_loss = 0.
    total_sample = 0
    with torch.no_grad():
        while True:
            user, item, _, seq = data.next_batch()  # (batch_size, seq_len), data.step += 1
            batch_size = user.size(0)
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            seq = seq.to(device)  # (batch_size, seq_len + 2)
            log_word_prob = model(user, item, seq[:, :-1])  # (batch_size,) vs. (batch_size, seq_len + 1, ntoken)
            loss = text_criterion(log_word_prob.view(-1, ntoken), seq[:, 1:].reshape((-1,)))

            text_loss += batch_size * loss.item()
            total_sample += batch_size

            if data.step == data.total_step:
                break
    return text_loss / total_sample


def generate(model, data):
    model.eval()
    idss_predict = []
    with torch.no_grad():
        while True:
            user, item, _, seq = data.next_batch()  # (batch_size, seq_len), data.step += 1
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            inputs = seq[:, :1].to(device)  # (batch_size, 1)
            hidden = None
            hidden_c = None
            ids = inputs
            for idx in range(args.words):
                # produce a word at each step
                if idx == 0:
                    hidden = model.encoder(user, item)
                    hidden_c = torch.zeros_like(hidden)
                    log_word_prob, hidden, hidden_c = model.decoder(inputs, hidden, hidden_c)  # (batch_size, 1, ntoken)
                else:
                    log_word_prob, hidden, hidden_c = model.decoder(inputs, hidden, hidden_c)  # (batch_size, 1, ntoken)
                word_prob = log_word_prob.squeeze().exp()  # (batch_size, ntoken)
                inputs = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1), pick the one with the largest probability
                ids = torch.cat([ids, inputs], 1)  # (batch_size, len++)
            ids = ids[:, 1:].tolist()  # remove bos
            idss_predict.extend(ids)

            if data.step == data.total_step:
                break
    return idss_predict

def load_predictor(args):
    print('loading model...')
    predictor = Att2SeqPredictor(
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


def main(args):
    predictor = load_predictor(args)
    model = predictor.model
    corpus = load_corpus(args)
    word2idx = corpus.word_dict.word2idx
    idx2word = corpus.word_dict.idx2word
    feature_set = corpus.feature_set
    test_data = Batchify(corpus.test, word2idx, 15, 128) # last 2 number: max seq words, bsize

    # Run on test data.
    test_loss = evaluate(model, test_data)
    print('=' * 89)
    print(now_time() + 'text ppl {:4.4f} | End of training'.format(math.exp(test_loss)))
    print(now_time() + 'Generating text')
    idss_predicted = generate(model, test_data)
    tokens_test = [ids2tokens(ids[1:], word2idx, idx2word) for ids in test_data.seq.tolist()]
    tokens_predict = [ids2tokens(ids, word2idx, idx2word) for ids in idss_predicted]
    BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
    print(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
    BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
    print(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))
    USR, USN = unique_sentence_percent(tokens_predict)
    print(now_time() + 'USR {:7.4f} | USN {:7}'.format(USR, USN))
    feature_batch = feature_detect(tokens_predict, feature_set)
    DIV = feature_diversity(feature_batch)  # time-consuming
    print(now_time() + 'DIV {:7.4f}'.format(DIV))
    FCR = feature_coverage_ratio(feature_batch, feature_set)
    print(now_time() + 'FCR {:7.4f}'.format(FCR))
    FMR = feature_matching_ratio(feature_batch, test_data.feature)
    print(now_time() + 'FMR {:7.4f}'.format(FMR))
    text_test = [' '.join(tokens) for tokens in tokens_test]
    text_predict = [' '.join(tokens) for tokens in tokens_predict]
    ROUGE = rouge_score(text_test, text_predict)  # a dictionary
    for (k, v) in ROUGE.items():
        print(now_time() + '{} {:7.4f}'.format(k, v))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Att2Seq (EACL\'17) without rating input')
    parser.add_argument('--data_path', type=str, default=None,
                        help='path for loading the pickle data')
    parser.add_argument('--index_dir', type=str, default=None,
                        help='load indexes')
    parser.add_argument('--model_weight_dir', type=str, default=None,
                        help='model weight dir')
    args = parser.parse_args()
    main(args)
    

