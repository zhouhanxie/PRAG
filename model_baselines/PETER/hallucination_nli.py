"""
script for using NLI model to detect hallucination
"""

import os
import torch 
import pickle
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from utils import DataLoader, ids2tokens
from collections import defaultdict

def load_corpus(args):
    print('loading corpus...')
    corpus = DataLoader(
        args.data_path, 
        args.index_dir, 
        20000
        )
    return corpus 

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    """https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main(args):

    # building historical reviews for each item
    corpus = load_corpus(args)
    print('building knowledge')
    item2reviews = defaultdict(list)
    for sample in corpus.test:
        item2reviews[sample['item']].append(' '.join(
            ids2tokens(sample['text'], corpus.word_dict.word2idx, corpus.word_dict.idx2word)
        ))
    

    print('loading mnli model')
    model_name = 'roberta-large-mnli'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.to(device)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0)
    data = pd.read_csv(os.path.join(args.output_dir, 'generated.csv'))

    all_entailed_pairs = []
    all_predicted = []
    for idx in tqdm(data.index[:args.max_test_sample]):
        predicted = data.at[idx, 'text_predicted']
        all_predicted.append(predicted)
        item = data.at[idx, 'item']
        sample = []
        for reference in item2reviews[item]:
            sample.append(reference+' </s></s> '+predicted)
        results = []
        for batch in chunks(sample, 32):
            result = classifier(batch)
            results += result 
        assert len(results) == len(sample), str(len(results))+' '+str(len(sample))
        
        entailed_pairs = []
        for idx, pred in enumerate(results):
            if pred['label'] == 'ENTAILMENT':
                entailed_pairs.append(sample[idx])
        all_entailed_pairs.append(entailed_pairs)

        


    
    num_hallucinated = 0
    
    for idx, res in enumerate(all_entailed_pairs):
        if len(res) == 0:
            num_hallucinated += 1
            
    print('rate hallucinated: ', num_hallucinated/args.max_test_sample)
    print('total sample: ', args.max_test_sample)

    print('generating output file...')
    with open(os.path.join(args.output_dir, 'hallucination_nli.out'), 'w') as ofp:
        for idx, r in enumerate(all_entailed_pairs):
            ofp.write(all_predicted[idx]+' @supported by@ '+str(r)+'\n')
            
        ofp.write(
            'rate hallucinated: '+str(num_hallucinated/args.max_test_sample)+'\n'
            )
        ofp.write(
            'total sample: '+ str(len(all_entailed_pairs))
            )
    
        

if __name__ == '__main__':
    from easydict import EasyDict
    main(
        EasyDict(
            {
                'data_path': '../nlg4rec_data/Yelp/reviews.pickle',
                'index_dir': '../nlg4rec_data/Yelp/1',
                'output_dir': './yelp_1/evaluations',
                'max_test_sample':10
            }
        )
    )