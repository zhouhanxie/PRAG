import os
import math
import torch
import argparse
import torch.nn as nn
import numpy as np
from module import PETER
from utils import rouge_score, bleu_score, DataLoader, Batchify, now_time, ids2tokens, unique_sentence_percent,  root_mean_square_error, mean_absolute_error, feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity


class PeterPredictor:
    """
    Thin wrapper around peter for inference.
    *written in a rush, might contain extra code blocks*
    """
    
    def __init__(self, 
                 model_path, 
                 dataloader_data_path, 
                 dataloader_index_dir,
                 dataloader_vocab_size,
                 use_feature=True,
                 words = 15,
                 device = torch.device('cuda')
                ):
        """
        Input:
            model path: path to .pt file
            dataloader args: see utils.DataLoader
            use_feature:
                weather to use features
            words:
                max sequence length
        """
        
        corpus = DataLoader(dataloader_data_path, dataloader_index_dir, dataloader_vocab_size)
        self.word2idx = corpus.word_dict.word2idx
        self.idx2word = corpus.word_dict.idx2word
        self.use_feature = use_feature
        self.device = device
        self.max_seq_words = words
        
        test_data = Batchify(corpus.test, self.word2idx, 15, batch_size=64)
        
        
        if use_feature:
            src_len = 2 + test_data.feature.size(1)  # [u, i, f]
        else:
            src_len = 2  # [u, i]

        tgt_len = words + 1  # added <bos> or <eos>
        ntokens = len(corpus.word_dict)
        nuser = len(corpus.user_dict)
        nitem = len(corpus.item_dict)
        pad_idx = self.word2idx['<pad>']
        
        self.ntokens = ntokens
        self.nll_loss = nn.NLLLoss(ignore_index=pad_idx)
        
        # load model
        model = PETER(True, src_len, tgt_len, pad_idx, nuser, nitem, ntokens, 512, 2, 2048, 2, 0.2).to(device)
        with open(model_path, 'rb') as f:
            model = torch.load(f, map_location='cpu').to(device)
        self.model = model
        
    def encode_with_unk(self, tok):
        if tok in self.word2idx:
            return self.word2idx[tok]
        else:
            return self.word2idx['<unk>']
    
    def wrap_with_padding(self, tok_list, seq_len=15):
        if len(tok_list) < seq_len:
            tok_list.append('<eos>')
        if len(tok_list) > seq_len:
            tok_list = tok_list[:seq_len]

        while len(tok_list) < seq_len:
            tok_list.append('<pad>')
        return [self.word2idx['<bos>']]+[self.encode_with_unk(t) for t in tok_list]# +[word2idx['<eos>']]

    def seq2language(self,list_of_id):
        return ' '.join([
            self.idx2word[idx] for idx in list_of_id if idx not in \
            {self.word2idx['<bos>'],self.word2idx['<eos>'],self.word2idx['<pad>']}
        ])
        
    
    
    def _inference(self, user, item, rating, seq, feature):
        self.model.eval()
        
        with torch.no_grad():  

            if self.use_feature:
                text = torch.cat([feature, seq[:-1]], 0)  # (src_len + tgt_len - 2, batch_size)
            else:
                text = seq[:-1]  # (src_len + tgt_len - 2, batch_size)
            log_word_prob, log_context_dis, rating_p, _ = self.model(user, item, text)  
            
        return log_word_prob, log_context_dis, rating_p
    
    def inference(self, user, item, rating, seq, feature): 
        
        device = self.device
        user = user.to(device)  
        item = item.to(device)
        rating = rating.to(device)
        feature = feature.t().to(device)  # (1, batch_size)
        seq = seq.t().to(device)
        log_word_prob, log_context_dis, rating_p = self._inference(user, item, rating, seq, feature)
        t_loss = self.nll_loss(log_word_prob.view(-1, self.ntokens), seq[1:].reshape((-1,)))
        ppl = math.exp(t_loss.item())
        return {
            'ppl':ppl,
            'rating':rating_p.item()
        }
    
    def tensorize(self, input_dict):
        user = torch.tensor([input_dict['userid']])
        item = torch.tensor([input_dict['itemid']])
        rating = torch.tensor([input_dict['rating']])
        seq = torch.tensor([self.wrap_with_padding(input_dict['review'].split())])
        feature = torch.tensor([[self.word2idx[input_dict['feature']]]])
        return user, item, rating, seq, feature
    
    def ppl_from_input(self, input_dict):
            return self.inference(
                *self.tensorize(input_dict)
            )['ppl']

    def generate(self, user, item, rating, seq, feature):
        """
        takes in tensorized u,i,r,s,f
        returns predicted text and rating for that instance.
        """
        predictor = self
        # Turn on evaluation mode which disables dropout.
        predictor.model.eval()
        idss_predict = []
        context_predict = []
        rating_predict = []
        
        with torch.no_grad():
            device = predictor.device
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            bos = seq[:, 0].unsqueeze(0).to(device)  # (1, batch_size)
            feature = feature.t().to(device)  # (1, batch_size)
            if predictor.use_feature:
                text = torch.cat([feature, bos], 0)  # (src_len - 1, batch_size)
            else:
                text = bos  # (src_len - 1, batch_size)
            start_idx = text.size(0)
            for idx in range(predictor.max_seq_words):
                # produce a word at each step
                if idx == 0:
                    log_word_prob, log_context_dis, rating_p, _ = predictor.model(user, item, text, False)  # (batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
                    rating_predict.append(rating_p.item())
                    # context = predict(log_context_dis, topk=predictor.max_seq_words)  # (batch_size, words)
                    # context_predict.extend(context.tolist())
                else:
                    log_word_prob, _, _, _ = predictor.model(user, item, text, False, False, False)  # (batch_size, ntoken)
                word_prob = log_word_prob.exp()  # (batch_size, ntoken)
                word_idx = torch.argmax(word_prob, dim=1)  # (batch_size,), pick the one with the largest probability
                text = torch.cat([text, word_idx.unsqueeze(0)], 0)  # (len++, batch_size)
            ids = text[start_idx:].t().tolist()  # (batch_size, seq_len)
            idss_predict.extend(ids)
            
        # rating
        predicted_rating = [(r, p) for (r, p) in zip(rating.tolist(), rating_predict)]
        
        # text
        tokens_predict = [ids2tokens(ids, predictor.word2idx, predictor.idx2word) for ids in idss_predict]
        text_predict = [' '.join(tokens) for tokens in tokens_predict]

        return text_predict[0], rating_predict[0]



def generate(predictor, user, item, rating, seq, feature):
    """
    takes in tensorized u,i,r,s,f
    returns predicted text and rating for that instance.
    """
    # Turn on evaluation mode which disables dropout.
    predictor.model.eval()
    idss_predict = []
    context_predict = []
    rating_predict = []
    
    with torch.no_grad():
        device = predictor.device
        user = user.to(device)  # (batch_size,)
        item = item.to(device)
        bos = seq[:, 0].unsqueeze(0).to(device)  # (1, batch_size)
        feature = feature.t().to(device)  # (1, batch_size)
        if predictor.use_feature:
            text = torch.cat([feature, bos], 0)  # (src_len - 1, batch_size)
        else:
            text = bos  # (src_len - 1, batch_size)
        start_idx = text.size(0)
        for idx in range(predictor.max_seq_words):
            # produce a word at each step
            if idx == 0:
                log_word_prob, log_context_dis, rating_p, _ = predictor.model(user, item, text, False)  # (batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
                rating_predict.append(rating_p.item())
                # context = predict(log_context_dis, topk=predictor.max_seq_words)  # (batch_size, words)
                # context_predict.extend(context.tolist())
            else:
                log_word_prob, _, _, _ = predictor.model(user, item, text, False, False, False)  # (batch_size, ntoken)
            word_prob = log_word_prob.exp()  # (batch_size, ntoken)
            word_idx = torch.argmax(word_prob, dim=1)  # (batch_size,), pick the one with the largest probability
            text = torch.cat([text, word_idx.unsqueeze(0)], 0)  # (len++, batch_size)
        ids = text[start_idx:].t().tolist()  # (batch_size, seq_len)
        idss_predict.extend(ids)
        
    # rating
    predicted_rating = [(r, p) for (r, p) in zip(rating.tolist(), rating_predict)]
    
    # text
    tokens_predict = [ids2tokens(ids, predictor.word2idx, predictor.idx2word) for ids in idss_predict]
    text_predict = [' '.join(tokens) for tokens in tokens_predict]

    return text_predict[0], rating_predict[0]