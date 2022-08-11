from utils import *
from transformers import GPT2Tokenizer

class PeplermfPredictor:
    
    def __init__(self, model_path, max_seq_len=20):
        bos = '<bos>'
        eos = '<eos>'
        pad = '<pad>'
        self.device = device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token=bos, eos_token=eos, pad_token=pad)
        self.model = torch.load(model_path, map_location='cpu').to(self.device)
        self.max_seq_len = max_seq_len
        
    def tensorize(self, input_dict):
        """
        tensorize a dict with user,item ids and string review
        """
        user = torch.tensor([input_dict['userid']])
        item = torch.tensor([input_dict['itemid']])
        rating = torch.tensor([input_dict['rating']])
        tokenized = self.tokenizer([input_dict['review']],  return_tensors='pt')
        seq, mask = tokenized['input_ids'], tokenized['attention_mask']

        return user, item, rating, seq, mask

    def _generate(self, user, item):
        """
        ultility, use generate()
        """
        # Turn on evaluation mode which disables dropout.
        self.model.eval()

        with torch.no_grad():

            user = user.to(self.device)  # (batch_size,)
            item = item.to(self.device)
            text = torch.tensor([[
                self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.bos_token
                )
            ]]).to(self.device)
            for idx in range(self.max_seq_len):
                # produce a word at each step
                outputs, rating_p= self.model(user, item, text, None)
                last_token = outputs.logits[:, -1, :]  # the last token, (batch_size, ntoken)
                word_prob = torch.softmax(last_token, dim=-1)
                token = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1), pick the one with the largest probability
                if token == self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.eos_token
                ):
                    break
                text = torch.cat([text, token], 1)  # (batch_size, len++)
            ids = text[:, 1:].tolist()  # remove bos, (batch_size, seq_len)

        out = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(ids[0])
        )
        return {
            'text_predicted':out,
            'rating_predicted':rating_p.item()
        }
    
    def _ppl(self, user, item, seq, mask):
        """
        ultility, use ppl()
        """
        self.model.eval()

        with torch.no_grad():

            user = user.to(self.device)  # (batch_size,)
            item = item.to(self.device)
            seq = seq.to(self.device)  # (batch_size, seq_len)
            mask = mask.to(self.device)
            outputs, rating_p = self.model(user, item, seq, mask)
            t_loss = outputs.loss.item()
            ppl = math.exp(t_loss)
            
        return ppl
    
    def generate(self, user, item):
        """
        generate reviwe given u,i
        """
        return self._generate(
            torch.tensor([user]),
            torch.tensor([item])
        )
    
    def ppl(self, user, item, text):
        """
        return perplexity given u,i,review
        """
        tokenized = self.tokenizer([text],  return_tensors='pt')
        seq, mask = tokenized['input_ids'], tokenized['attention_mask']
        user = torch.tensor([user])
        item = torch.tensor([item])
        
        return self._ppl(
            torch.tensor([user]), 
            torch.tensor([item]), 
            seq, 
            mask
        )
        