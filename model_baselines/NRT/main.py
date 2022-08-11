import os
import math
import torch
import argparse
import torch.nn as nn
from module import NRT
from utils import rouge_score, bleu_score, DataLoader, Batchify, now_time, ids2tokens, unique_sentence_percent, \
    root_mean_square_error, mean_absolute_error, feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity
from gputils import occupy_mem 
import launcher
occupy_mem(cuda_device=str(launcher.load_gpuid()), amount='auto')


parser = argparse.ArgumentParser(description='NRT (SIGIR\'17) without review prediction task')
parser.add_argument('--data_path', type=str, default=None,
                    help='path for loading the pickle data')
parser.add_argument('--index_dir', type=str, default=None,
                    help='load indexes')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of embeddings')
parser.add_argument('--nhid', type=int, default=400,
                    help='number of hidden units')
parser.add_argument('--nlayers', type=int, default=4,
                    help='number of layers for rating prediction')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=200,
                    help='batch size')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--checkpoint', type=str, default='./nrt/',
                    help='directory to save the final model')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--vocab_size', type=int, default=20000,
                    help='keep the most frequent words in the vocabulary')
parser.add_argument('--endure_times', type=int, default=5,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('--l2_reg', type=float, default=0,
                    help='L2 regularization (0.0001 as reported in the paper would cause identical sentences)')
parser.add_argument('--text_reg', type=float, default=1.0,
                    help='regularization on text generation task')
parser.add_argument('--rating_reg', type=float, default=1.0,
                    help='regularization on rating prediction task')
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
model = NRT(nuser, nitem, ntoken, args.emsize, args.nhid, args.nlayers, corpus.max_rating, corpus.min_rating).to(device)
text_criterion = nn.NLLLoss(ignore_index=pad_idx)  # ignore the padding when computing loss
rating_criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#optimizer = torch.optim.Adadelta(model.parameters())  # lr is optional to Adadelta

###############################################################################
# Training code
###############################################################################


def train(data):
    model.train()
    text_loss = 0.
    rating_loss = 0.
    total_sample = 0
    while True:
        user, item, rating, seq = data.next_batch()  # (batch_size, seq_len), data.step += 1
        batch_size = user.size(0)
        user = user.to(device)  # (batch_size,)
        item = item.to(device)
        rating = rating.to(device)
        seq = seq.to(device)  # (batch_size, seq_len + 2)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
        rating_p, log_word_prob = model(user, item, seq[:, :-1])  # (batch_size,) vs. (batch_size, seq_len + 1, ntoken)
        r_loss = rating_criterion(rating_p, rating)
        t_loss = text_criterion(log_word_prob.view(-1, ntoken), seq[:, 1:].reshape((-1,)))
        l2_loss = torch.cat([x.view(-1) for x in model.parameters()]).pow(2.).sum()
        loss = args.text_reg * t_loss + args.rating_reg * r_loss + args.l2_reg * l2_loss
        loss.backward()
        optimizer.step()

        text_loss += batch_size * t_loss.item()
        rating_loss += batch_size * r_loss.item()
        total_sample += batch_size

        if data.step == data.total_step:
            break
    return text_loss / total_sample, rating_loss / total_sample


def evaluate(data):
    model.eval()
    text_loss = 0.
    rating_loss = 0.
    total_sample = 0
    with torch.no_grad():
        while True:
            user, item, rating, seq = data.next_batch()  # (batch_size, seq_len), data.step += 1
            batch_size = user.size(0)
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            rating = rating.to(device)
            seq = seq.to(device)  # (batch_size, seq_len + 2)
            rating_p, log_word_prob = model(user, item, seq[:, :-1])  # (batch_size,) vs. (batch_size, seq_len + 1, ntoken)
            r_loss = rating_criterion(rating_p, rating)
            t_loss = text_criterion(log_word_prob.view(-1, ntoken), seq[:, 1:].reshape((-1,)))

            text_loss += batch_size * t_loss.item()
            rating_loss += batch_size * r_loss.item()
            total_sample += batch_size

            if data.step == data.total_step:
                break
    return text_loss / total_sample, rating_loss / total_sample


def generate(data):
    model.eval()
    idss_predict = []
    rating_predict = []
    with torch.no_grad():
        while True:
            user, item, _, seq = data.next_batch()  # (batch_size, seq_len), data.step += 1
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            inputs = seq[:, :1].to(device)  # (batch_size, 1)
            hidden = None
            ids = inputs
            for idx in range(args.words):
                # produce a word at each step
                if idx == 0:
                    rating_p, hidden = model.encoder(user, item)
                    rating_predict.extend(rating_p.tolist())
                    log_word_prob, hidden = model.decoder(inputs, hidden)  # (batch_size, 1, ntoken)
                else:
                    log_word_prob, hidden = model.decoder(inputs, hidden)  # (batch_size, 1, ntoken)
                word_prob = log_word_prob.squeeze().exp()  # (batch_size, ntoken)
                inputs = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1), pick the one with the largest probability
                ids = torch.cat([ids, inputs], 1)  # (batch_size, len++)
            ids = ids[:, 1:].tolist()  # remove bos
            idss_predict.extend(ids)

            if data.step == data.total_step:
                break
    return idss_predict, rating_predict


# Loop over epochs.
best_val_loss = float('inf')
endure_count = 0
for epoch in range(1, args.epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))
    train_t_loss, train_r_loss = train(train_data)
    print(now_time() + 'text ppl {:4.4f} | rating loss {:4.4f} | total loss {:4.4f} on train'.format(
        math.exp(train_t_loss), train_r_loss, train_t_loss + train_r_loss))
    val_t_loss, val_r_loss = evaluate(val_data)
    val_loss = val_t_loss + val_r_loss
    print(now_time() + 'text ppl {:4.4f} | rating loss {:4.4f} | total loss {:4.4f} on validation'.format(
        math.exp(val_t_loss), val_r_loss, val_loss))
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
test_t_loss, test_r_loss = evaluate(test_data)
print('=' * 89)
print(now_time() + 'text ppl {:4.4f} | rating loss {:4.4f} | total loss {:4.4f} on test | End of training'.format(
        math.exp(test_t_loss), test_r_loss, test_t_loss + test_r_loss))
print(now_time() + 'Generating text')
idss_predicted, rating_predicted = generate(test_data)
# rating
predicted_rating = [(r, p) for (r, p) in zip(test_data.rating.tolist(), rating_predicted)]
RMSE = root_mean_square_error(predicted_rating, corpus.max_rating, corpus.min_rating)
print(now_time() + 'RMSE {:7.4f}'.format(RMSE))
MAE = mean_absolute_error(predicted_rating, corpus.max_rating, corpus.min_rating)
print(now_time() + 'MAE {:7.4f}'.format(MAE))
# text
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
