import os
import math
import torch
import argparse
import torch.nn as nn
from module import Att2Seq
from utils import rouge_score, bleu_score, DataLoader, Batchify, now_time, ids2tokens, unique_sentence_percent, \
    feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity

from gputils import occupy_mem 
import launcher
occupy_mem(cuda_device=str(launcher.load_gpuid()), amount=4000)


parser = argparse.ArgumentParser(description='Att2Seq (EACL\'17) without rating input')
parser.add_argument('--data_path', type=str, default=None,
                    help='path for loading the pickle data')
parser.add_argument('--index_dir', type=str, default=None,
                    help='load indexes')
parser.add_argument('--emsize', type=int, default=64,
                    help='size of user/item embeddings')
parser.add_argument('--nhid', type=int, default=512,
                    help='number of hidden units')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of LSTM layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=5.0,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=50,
                    help='batch size')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--checkpoint', type=str, default='./att2seq/',
                    help='directory to save the final model')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--vocab_size', type=int, default=20000,
                    help='keep the most frequent words in the vocabulary')
parser.add_argument('--endure_times', type=int, default=5,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('--words', type=int, default=15,
                    help='number of words to generate for each sample')
args = parser.parse_args()

if args.data_path is None:
    parser.error('--data_path should be provided for loading data')
if args.index_dir is None:
    parser.error('--index_dir should be provided for loading data splits')

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

if torch.cuda.is_available():
    if not args.cuda:
        print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
device = torch.device('cuda'+':'+str(launcher.load_gpuid()) if args.cuda else 'cpu')
print('your device: ', device)

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)
model_path = os.path.join(args.checkpoint, 'model.pt')
prediction_path = os.path.join(args.checkpoint, args.outf)

###############################################################################
# Load data
###############################################################################

print(now_time() + 'Loading data')
corpus = DataLoader(args.data_path, args.index_dir, args.vocab_size)
word2idx = corpus.word_dict.word2idx
idx2word = corpus.word_dict.idx2word
feature_set = corpus.feature_set
train_data = Batchify(corpus.train, word2idx, args.words, args.batch_size, shuffle=True)
val_data = Batchify(corpus.valid, word2idx, args.words, args.batch_size)
test_data = Batchify(corpus.test, word2idx, args.words, args.batch_size)

###############################################################################
# Build the model
###############################################################################

nuser = len(corpus.user_dict)
nitem = len(corpus.item_dict)
ntoken = len(corpus.word_dict)
pad_idx = word2idx['<pad>']
model = Att2Seq(nuser, nitem, ntoken, args.emsize, args.nhid, args.dropout, args.nlayers).to(device)
text_criterion = nn.NLLLoss(ignore_index=pad_idx)  # ignore the padding when computing loss
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=0.002, alpha=0.95)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97)

###############################################################################
# Training code
###############################################################################


def train(data):
    model.train()
    text_loss = 0.
    total_sample = 0
    while True:
        user, item, _, seq = data.next_batch()  # (batch_size, seq_len), data.step += 1
        batch_size = user.size(0)
        user = user.to(device)  # (batch_size,)
        item = item.to(device)
        seq = seq.to(device)  # (batch_size, seq_len + 2)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
        log_word_prob = model(user, item, seq[:, :-1])  # (batch_size,) vs. (batch_size, seq_len + 1, ntoken)
        loss = text_criterion(log_word_prob.view(-1, ntoken), seq[:, 1:].reshape((-1,)))
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        text_loss += batch_size * loss.item()
        total_sample += batch_size

        if data.step == data.total_step:
            break
    return text_loss / total_sample


def evaluate(data):
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


def generate(data):
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


# Loop over epochs.
best_val_loss = float('inf')
endure_count = 0
for epoch in range(1, args.epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))
    train_loss = train(train_data)
    print(now_time() + 'text ppl {:4.4f} on train'.format(math.exp(train_loss)))
    val_loss = evaluate(val_data)
    print(now_time() + 'text ppl {:4.4f} on validation'.format(math.exp(val_loss)))
#    if epoch > 10:
        # Anneal the learning rate
#        scheduler.step()
#        print(now_time() + 'Learning rate set to {:2.8f}'.format(scheduler.get_last_lr()[0]))
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        with open(model_path, 'wb') as f:
            torch.save(model, f)
    else:
        endure_count += 1
        print(now_time() + 'Endured {} time(s)'.format(endure_count))
        if endure_count == args.endure_times:
            print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
            break

# Load the best saved model.
with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)


# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print(now_time() + 'text ppl {:4.4f} | End of training'.format(math.exp(test_loss)))
print(now_time() + 'Generating text')
idss_predicted = generate(test_data)
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
text_out = ''
for (real, fake) in zip(text_test, text_predict):
    text_out += '{}\n{}\n\n'.format(real, fake)
with open(prediction_path, 'w', encoding='utf-8') as f:
    f.write(text_out)
print(now_time() + 'Generated text saved to ({})'.format(prediction_path))
