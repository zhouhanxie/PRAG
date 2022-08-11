import os
import torch 
import pandas as pd
from collections import Counter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


def main(args):

    model_name = 'roberta-large-mnli'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # tokenizer.to(device)
    # model.to(device)


    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0)


    data = pd.read_csv(os.path.join(args.output_dir, 'generated.csv'))

    results = []
    for idx in tqdm(data.index[:args.max_test_sample]):
        real = data.at[idx, 'review']
        predicted = data.at[idx, 'text_predicted']
        sample = [real+' </s></s> '+predicted]
        result = classifier(sample)[0]
        result['ground_truth'] = real
        result['generated'] = predicted
        results.append(result)

    print('total: ', len(results))
    print('distr: ', dict(Counter([i['label'] for i in results])))

    with open(os.path.join(args.output_dir, 'entailment.out'), 'w') as ofp:
        for r in results:
            ofp.write(str(r)+'\n')
        ofp.write(
            'distr: ' + str(dict(Counter([i['label'] for i in results])))
            )
        




