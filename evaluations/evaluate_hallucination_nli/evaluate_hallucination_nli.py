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
from collections import defaultdict
from transformers import GPT2Tokenizer


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    """https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]  

def main(args):
    
    # load data
    dat = pd.read_csv(
        args.input_file
    )

    # load the entailment model
    print('loading mnli model')
    model_name = 'roberta-large-mnli'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0)




    # run entailment check
    tot_entail = 0
    tot_neutral = 0
    tot_contradiction = 0
    for datum in tqdm(dat.to_dict('records')):
        generated = datum['generated']
        evidence = eval(datum['evidence'])

        entail_pipeline_input = []
        for reference in  evidence:
            entail_pipeline_input.append(reference+' </s></s> '+generated)

        entail_pipeline_output = []
        for batch in chunks(entail_pipeline_input, 32):
            result = classifier(batch)
            entail_pipeline_output += result 

        is_entail = 0
        for entailment_decision in entail_pipeline_output:
            if entailment_decision['label'] == 'ENTAILMENT':
                is_entail = 1
                break
        tot_entail += is_entail

    
    print('entailment: ', tot_entail, 'out of', len(dat))
    print('ratio')
    print(tot_entail/len(dat))

    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default=None,
                        help='input file, must have key field generated and evidence')
    args = parser.parse_args()
    main(args)