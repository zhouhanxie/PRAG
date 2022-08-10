import nltk
from nltk import ngrams
from collections import Counter
from nltk.probability import FreqDist
import numpy as np
import pandas as pd

def distinct_n_sentence_level(sentence, n):
    """
    Compute distinct-N for a single sentence.
    :param sentence: a list of words.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)


def distinct_n_corpus_level(sentences, n):
    """
    Compute average distinct-N of a list of sentences (the corpus).
    :param sentences: a list of sentence.
    :param n: int, ngram.
    :return: float, the average value.
    """
    return sum(distinct_n_sentence_level(sentence, n) for sentence in sentences) / len(sentences)

def get_diversity(corpus):
    """
    coprus:
     :list[str]
    return:
        none
    prints:
        d-1,2,3; ENTR 1,2,3
    """
    
    entr_scores = []
    for gram_size in [1,2,3]:
        all_grams = []
        for sent in corpus:
            all_grams += list(ngrams(sent.split(), gram_size))

        fdist = FreqDist(Counter(all_grams))

        entr = 0
        for x in fdist.keys():
            p_x = fdist.freq(x)
            entr += p_x*np.log2(p_x)

        print('ENTR ', gram_size)
        print(-entr)
        print()
        print('DISTINCT ', gram_size)
        print(distinct_n_corpus_level(corpus, gram_size))
        print('*'*40)
        
        entr_scores.append(-entr)
        
    print('geometric mean of ENTR scores')
    print(np.mean(entr_scores))
    print()
    print('unique sentence ratio (USR)')
    print(len(set(corpus))/len(list(corpus)))


def main(args):
    file_type = args.input_file.split('.')[-1]
    if file_type == 'txt':
        with open(args.input_file) as f:
            dat = [l.strip() for l in f.readlines()][:10000]
    elif file_type == 'csv':
        dat = pd.read_csv(args.input_file)[args.csv_key]
    else:
        raise TypeError 
        
    
    out_file = args.input_file+'.diversity'
    import sys
    sys.stdout = open(out_file, 'w')
    get_diversity(dat)
    sys.stdout.close()

    

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(description='Personalized Retriever model inference')
    parser.add_argument('--input_file', type=str, default=None,
                        help='path for the input file')
    parser.add_argument('--csv_key', type=str, default='generated',
                        help='key for the column to be checked, if file is csv')
    args = parser.parse_args()

    main(args)