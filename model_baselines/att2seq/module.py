import torch
import torch.nn as nn
import torch.nn.functional as func


class MLPEncoder(nn.Module):
    def __init__(self, nuser, nitem, emsize, hidden_size, nlayers):
        super(MLPEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)
        self.encoder = nn.Linear(emsize * 2, hidden_size * nlayers)
        self.tanh = nn.Tanh()

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.encoder.bias.data.zero_()

    def forward(self, user, item):  # (batch_size,)
        u_src = self.user_embeddings(user)  # (batch_size, emsize)
        i_src = self.item_embeddings(item)
        ui_concat = torch.cat([u_src, i_src], 1)  # (batch_size, emsize * 2)
        hidden = self.tanh(self.encoder(ui_concat))  # (batch_size, hidden_size * nlayers)
        state = hidden.reshape((-1, self.nlayers, self.hidden_size)).permute(1, 0, 2).contiguous()  # (num_layers, batch_size, hidden_size)
        return state


class LSTMDecoder(nn.Module):
    def __init__(self, ntoken, emsize, hidden_size, num_layers, dropout):
        super(LSTMDecoder, self).__init__()
        self.word_embeddings = nn.Embedding(ntoken, emsize)
        self.lstm = nn.LSTM(emsize, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.08
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()

    def forward(self, seq, ht, ct):  # seq: (batch_size, seq_len), ht & ct: (nlayers, batch_size, hidden_size)
        seq_emb = self.word_embeddings(seq)  # (batch_size, seq_len, emsize)
        output, (ht, ct) = self.lstm(seq_emb, (ht, ct))  # (batch_size, seq_len, hidden_size) vs. (nlayers, batch_size, hidden_size)
        decoded = self.linear(output)  # (batch_size, seq_len, ntoken)
        return func.log_softmax(decoded, dim=-1), ht, ct


class Att2Seq(nn.Module):
    def __init__(self, nuser, nitem, ntoken, emsize, hidden_size, dropout, num_layers=2):
        super(Att2Seq, self).__init__()
        self.encoder = MLPEncoder(nuser, nitem, emsize, hidden_size, num_layers)
        self.decoder = LSTMDecoder(ntoken, hidden_size, hidden_size, num_layers, dropout)

    def forward(self, user, item, seq):  # (batch_size,) vs. (batch_size, seq_len)
        h0 = self.encoder(user, item)  # (num_layers, batch_size, hidden_size)
        c0 = torch.zeros_like(h0)
        log_word_prob, _, _ = self.decoder(seq, h0, c0)
        return log_word_prob  # (batch_size, seq_len, ntoken)
