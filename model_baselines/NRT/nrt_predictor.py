import os
import math
import torch
import argparse
import torch.nn as nn
import numpy as np
from module import NRT
from utils import *

class NRTPredictor:
    """
    Thin wrapper around Att2Seq for inference.
    *written in a rush, might contain extra code blocks*
    """
    
    def __init__(self, 
                 model_path, 
                 dataloader_data_path, 
                 dataloader_index_dir,
                 dataloader_vocab_size,
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
        self.device = device
        self.max_seq_words = words
        
        test_data = Batchify(corpus.test, self.word2idx, 15, batch_size=64)
        
        
#         if use_feature:
#             src_len = 2 + test_data.feature.size(1)  # [u, i, f]
#         else:
#             src_len = 2  # [u, i]

#         tgt_len = words + 1  # added <bos> or <eos>
        ntokens = len(corpus.word_dict)
        nuser = len(corpus.user_dict)
        nitem = len(corpus.item_dict)
        pad_idx = self.word2idx['<pad>']
        self.ntokens = ntokens
        
        self.ntokens = ntokens
        self.nll_loss = nn.NLLLoss(ignore_index=pad_idx)
        
        # load model
        emsize = 300
        nhid = 400
        dropout = 0.2
        nlayers = 4
        model = NRT(
            nuser, nitem, ntokens, emsize, nhid, nlayers, corpus.max_rating, corpus.min_rating
        ).to(device)
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
        
    
    
    def _generate(self, user, item):
        self.model.eval()
        with torch.no_grad():
            device = self.device
            user = user.to(device)  
            item = item.to(device)
            inputs = torch.tensor([[self.word2idx['<bos>']]])[:, :1].to(device)
            hidden = None
            hidden_c = None
            ids = inputs
            for idx in range(self.max_seq_words):
                # produce a word at each step
                if idx == self.word2idx['<bos>']:
                    rating_p, hidden = self.model.encoder(user, item)
                    log_word_prob, hidden = self.model.decoder(inputs, hidden)  # (batch_size, 1, ntoken)
                else:
                    log_word_prob, hidden = self.model.decoder(inputs, hidden)  # (batch_size, 1, ntoken)
                word_prob = log_word_prob.squeeze().exp()  # (batch_size, ntoken)
                inputs = torch.tensor([[torch.argmax(word_prob, dim=-1)]]).to(device)  # (batch_size, 1), pick the one with the largest probability
                ids = torch.cat([ids, inputs], 1)  # (batch_size, len++)
            generated = self.seq2language(ids.flatten().tolist())  
        return {
            'text_predicted':generated, 
            'rating_predicted':rating_p.item()
        }
    
    def _ppl(self, user, item, seq): 
        self.model.eval()
        with torch.no_grad():
            device = self.device
            user = user.to(device)  
            item = item.to(device)
            seq = seq.to(device)
            rating_p, log_word_prob = self.model(user, item, seq[:, :-1])
            t_loss = self.nll_loss(log_word_prob.view(-1, self.ntokens), seq[:, 1:].reshape((-1,)))
            ppl = math.exp(t_loss.item())
        return ppl
    
    def ppl(self, user, item, seq):
        u,i,s = self.tensorize(
            {
                'userid':user,
                'itemid':item,
                'review':seq
            }
        )
        return self._ppl(u,i,s)
    
    def generate(self, user, item):
        u,i,_ = self.tensorize(
            {
                'userid':user,
                'itemid':item,
                'review':'stub'
            }
        )
        return self._generate(u,i)
        
    
    def tensorize(self, input_dict):
        user = torch.tensor([input_dict['userid']])
        item = torch.tensor([input_dict['itemid']])
        seq = torch.tensor([self.wrap_with_padding(input_dict['review'].split())])
        return user, item, seq


if __name__ == '__main__':
    predictor = NRTPredictor(
        'NRT/tripadvisor/model.pt',
        '../nlg4rec_data/TripAdvisor/reviews.pickle',
        '../nlg4rec_data/TripAdvisor/1/',
        20000
    )
    predictor.ppl(3,0,'hotel is darn great')
    predictor.generate(3,0)

    # done, inconsistent labels: 5724