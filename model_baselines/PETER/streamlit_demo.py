import os
import math
import torch
import argparse
import joblib
import streamlit as st
import numpy as np
import torch.nn as nn
from module import PETER
from utils import rouge_score, bleu_score, DataLoader, Batchify, \
now_time, ids2tokens, unique_sentence_percent,  root_mean_square_error, \
mean_absolute_error, feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity



def setup():
    """
    Squashed all the preparation code into this function
    """
    model_path = os.path.join('model/tripadvisorf_backup/model.pt')
    
    corpus = joblib.load('datasample_for_streamlit.joblib')
    batch_size=64
    word2idx = corpus.word_dict.word2idx
    idx2word = corpus.word_dict.idx2word
    feature_set = corpus.feature_set
    # train_data = Batchify(corpus.train, word2idx, 15, batch_size, shuffle=True)
    # val_data = Batchify(corpus.valid, word2idx, 15, batch_size)
    test_data = Batchify(corpus.test, word2idx, 15, batch_size)
    use_feature=True
    words = 15

    if use_feature:
        src_len = 2 + test_data.feature.size(1)  # [u, i, f]
    else:
        src_len = 2  # [u, i]

    tgt_len = words + 1  # added <bos> or <eos>
    ntokens = len(corpus.word_dict)
    nuser = len(corpus.user_dict)
    nitem = len(corpus.item_dict)
    pad_idx = word2idx['<pad>']
    # test_data = Batchify(corpus.test, word2idx, words, batch_size)
    text_criterion = nn.NLLLoss(ignore_index=pad_idx)
    device = torch.device('cuda')
    # print('running on ', device)
    # model = None
    model = PETER(True, src_len, tgt_len, pad_idx, nuser, nitem, ntokens, 512, 2, 2048, 2, 0.2).to(device)
    # Load the best saved model.
    with open(model_path, 'rb') as f:
        model = torch.load(f).to(device)
    
    
    return model, test_data, word2idx, idx2word, text_criterion, ntokens, nuser, nitem, pad_idx, text_criterion


def main():
    model, test_data, word2idx, idx2word, text_criterion, ntokens, nuser, nitem, pad_idx, text_criterion = setup()
    use_feature = True
    device = torch.device('cuda')

    corpus = joblib.load('datasample_for_streamlit.joblib')
    test_data = corpus.test 
    word2idx = corpus.word_dict.word2idx
    idx2word = corpus.word_dict.idx2word

    # helpers
    def encode_with_unk(tok):
        if tok in word2idx:
            return word2idx[tok]
        else:
            return word2idx['<unk>']
    
    def wrap_with_padding(tok_list, seq_len=15):
        if len(tok_list) < seq_len:
            tok_list.append('<eos>')
        if len(tok_list) > seq_len:
            tok_list = tok_list[:seq_len]
        
        while len(tok_list) < seq_len:
            tok_list.append('<pad>')
        return [word2idx['<bos>']]+[encode_with_unk(t) for t in tok_list]# +[word2idx['<eos>']]

    def encode_seq(list_of_sent):
        tokens = [wrap_with_padding(s.split()) for s in list_of_sent]
        return tokens

    def seq2language(list_of_id):
        return ' '.join([
            idx2word[idx] for idx in list_of_id if idx not in \
            {word2idx['<bos>'],word2idx['<eos>'],word2idx['<pad>']}
        ])


    def batch_one(user, item, rating, seq, feature):
        """
        batch single items into s psuedobatch
        """
        pass

    def _ppl(user, item, seq, feature):
        if use_feature:
            text = torch.cat([feature, seq[:-1]], 0)  # (src_len + tgt_len - 2, batch_size)
        else:
            text = seq[:-1]  # (src_len + tgt_len - 2, batch_size)
        log_word_prob, log_context_dis, rating_p, _ = model(user, item, text)  # (tgt_len, batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
        t_loss = text_criterion(log_word_prob.view(-1, ntokens), seq[1:].reshape((-1,)))
        return math.exp(t_loss.item())

    def sample(test_idx, test_corpus, return_dict=False):
        
        
        sample = test_corpus[test_idx]

        if return_dict:
            return sample

        return sample['user'], sample['item'], sample['rating'], seq2language(sample['text']), idx2word[sample['feature']]

    def to_readable(user, item, rating, seq, feature):
        out = {
            'userid':user,
            'itemid':item,
            'rating':rating,
            'feature':feature,
            'review':seq
        }
        return out

    def ppl(model, user, item, rating, seq, feature):
        
        model.eval()
        with torch.no_grad():  
            user = user.to(device)  
            item = item.to(device)
            rating = rating.to(device)
            feature = feature.t().to(device)  # (1, batch_size)
            seq = seq.t().to(device)
            
            return _ppl(user, item, seq, feature)

    def ppl_from_input(model, input_dict):
        return ppl(
            model = model, 
            user = torch.tensor([input_dict['userid']]), 
            item = torch.tensor([input_dict['itemid']]), 
            rating = torch.tensor([input_dict['rating']]), 
            seq = torch.tensor([wrap_with_padding(input_dict['review'].split())]), 
            feature = torch.tensor([[word2idx[input_dict['feature']]]])
        )

    
    

    st.title("Interactive Demo For Breaking PETER")
    st.markdown('PETER is a I->OR model that jointly generates review and conduct rating estimation.', unsafe_allow_html=False)


    if 'current_datapoint' in st.session_state:
        readable_results = st.session_state['current_datapoint']
    else:
        user, item, rating, seq, feature = sample(5, test_data)
        seq = seq2language(wrap_with_padding(seq.split()))
        readable_results = to_readable(user, item, rating, seq, feature)
        st.session_state['current_datapoint'] = readable_results


    if st.button('Resample'):
        new_idx = np.random.randint(low=0, high=1000, size=None, dtype=int)
        user, item, rating, seq, feature = sample(new_idx, test_data)
        seq = seq2language(wrap_with_padding(seq.split()))
        readable_results = to_readable(user, item, rating, seq, feature)
        st.session_state['current_datapoint'] = readable_results
        st.write(readable_results)
        st.write(
            'ppl: ',
            ppl_from_input(
                model, 
                readable_results
                )
        )
    else:
        st.write(readable_results)
        st.write(
            'ppl: ',
            ppl_from_input(
                model, 
                readable_results
                )
        )
    
    

    input_str = st.text_input('Write perturbed input here', '')
    if isinstance( eval(input_str.strip()), dict):
        st.write('input received')
    else:
        st.write('input must be dict')

    # userid_input = st.number_input('UserId', min_value=0, value=readable_results['userid'], step=1)
    # itemid_input = st.number_input('ItemId', min_value=0, value=readable_results['itemid'], step=1)
    # avail_ratings = [1,2,3,4,5]
    # rating_input = st.selectbox('Rating',avail_ratings, index=avail_ratings.index(readable_results['rating']))
    # feature_input = st.text_input('Feature', readable_results['feature'])
    # review_input = st.text_area("Review (15 tokens)", readable_results['review'])

    if st.button('Compute ppl'):
        # input_str = seq2language(wrap_with_padding(input_str.split()))
        # # st.write(input_str)
        # new_results = readable_results.copy()
        # new_results['review'] = input_str
        new_results = eval(input_str)
        st.write(
            input_str+': '+str(ppl_from_input(
                model, 
                new_results
                ))
        )
    else:
        st.write('click to compute ppl')
    

if __name__ == "__main__":
    main()