import torch
import transformers
from transformers.models.bert.modeling_bert import (
    BertSelfAttention,
    BertSelfOutput,
    BertAttention,
    BertIntermediate,
    BertOutput,
    BertLayer,
    BertEncoder,
    BertPooler,
    BertPredictionHeadTransform,
    BertLMPredictionHead,
    BertPreTrainedModel,
    BertModel,
    BERT_START_DOCSTRING
)
import torch.nn as nn
from packaging import version
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)


class UncastableEmbedding(torch.nn.Embedding):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warned=False
        self.output_device = torch.device('cpu')
        
    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if not self.warned:
            print('[class UncastableEmbedding]: warning, cannot do dtype, device cast! (modeling_bertwithooutpositionembedding.py)')
            self.warned=True
        self.output_device = device
        return self
            
    def cuda(self, *args, **kwargs):
        if not self.warned:
            print('[class UncastableEmbedding]: warning, cannot do dtype, device cast! (modeling_bertwithooutpositionembedding.py)')
            self.warned=True
        self.output_device = torch.device('cuda')
        return self

    def cpu(self, *args, **kwargs):
        if not self.warned:
            print('[class UncastableEmbedding]: warning, cannot do dtype, device cast! (modeling_bertwithooutpositionembedding.py)')
            self.warned=True
        self.output_device = torch.device('cpu')

    def _apply(self, fn):
        # this is a recursive call fn object
        # that pytorch passes in the "to" method 
        if fn.__name__ == 'convert':
            if not self.warned:
                print('[class UncastableEmbedding]: warning, cannot do dtype, device cast via _apply (modeling_bertwithooutpositionembedding.py)')
                self.warned=True
            victim = torch.zeros(1)
            victim = fn(victim)
            self.output_device = victim.device 
        else:
            torch.nn.Embedding._apply(self, fn)

    def __call__(self, *args, **kwargs):

        return torch.nn.Embedding.__call__(self, *args, **kwargs).to(self.output_device)
        


class UnpositionedEmbeddings(torch.nn.Module):
    """
    Document Embedding that differs from a standard BERT only in that it
    constructs the embeddings from word, and token_type embeddings. (no position)
    """

    def __init__(self, config):
        super().__init__()
        
        self.force_word_embeddings_on_cpu = config.force_word_embeddings_on_cpu
        if config.force_word_embeddings_on_cpu:
            self.word_embeddings =UncastableEmbedding(
                config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
            )
        else:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
            
        if config.freeze_word_embeddings:
            for param in self.word_embeddings.parameters():
                param.requires_grad = False
        
        
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # useful for injecting embeddings (put in pointers of your embeddings)
        # these buffers will be poped one by one during forward
        self.external_embeddings = []
        self.concatenative_embeddings = []
        self.additive_embeddings = []
        
        

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=None
    ):  
            
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        if inputs_embeds == None:
            if self.force_word_embeddings_on_cpu:
                inputs_embeds = self.word_embeddings(input_ids.cpu()).to(input_ids.device)
            else:
                inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        while len(self.external_embeddings) > 0:
            next_emb = self.external_embeddings.pop()
            embeddings =  embeddings * next_emb

        while len(self.concatenative_embeddings) > 0:
            next_emb = self.concatenative_embeddings.pop()
            embeddings = torch.cat([next_emb, embeddings], dim=1) #(bsize, max_seq_len, n_hid), we cat to sequence

        while len(self.additive_embeddings) > 0:
            next_emb = self.additive_embeddings.pop()
            embeddings = embeddings + next_emb

        return embeddings

    
        
    
class BertWithoutPositionEmbeddingModel(transformers.BertModel):
    
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.embeddings = UnpositionedEmbeddings(config)
        
        
    def get_pooled_representation(self, input_ids, token_type_ids, attention_mask, **kwargs):
        model_output = self(
            input_ids = input_ids,
            token_type_ids = token_type_ids,
            attention_mask = attention_mask
        )
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_hidden_states_and_attention(self, input_ids, token_type_ids, attention_mask, **kwargs):
        model_output = self(
            input_ids = input_ids,
            token_type_ids = token_type_ids,
            attention_mask = attention_mask
        )
        return model_output[0], attention_mask


if __name__ == '__main__':
    from transformers import BertModel, BertConfig
    configuration = BertConfig(
        vocab_size = 5000,
        hidden_size = 384,
        num_hidden_layers = 1,
        num_attention_heads = 1,
        intermediate_size = 768,
        hidden_act = 'gelu',
        hidden_dropout_prob = 0.1,
        attention_probs_dropout_prob = 0.1,
        max_position_embeddings = 0,
        type_vocab_size = 2,
        initializer_range = 0.02,
        layer_norm_eps = 1e-12,
        pad_token_id = 0,
        use_cache = True,
        classifier_dropout = None,
        freeze_word_embeddings=True,
        force_word_embeddings_on_cpu=True
    )


    model = BertWithoutPositionEmbeddingModel(configuration)