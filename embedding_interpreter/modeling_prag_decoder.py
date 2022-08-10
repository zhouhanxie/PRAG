"""Personalized Causal Rag Model"""
from collections import OrderedDict
from typing import Any, List, Mapping, Optional

from transformers import PreTrainedTokenizer, TensorType, is_torch_available

from transformers.configuration_utils import PretrainedConfig
from transformers.onnx import OnnxConfigWithPast, PatchingSpec
# from transformers.utils import logging

# logger = logging.get_logger(__name__)

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple


import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


if version.parse(torch.__version__) >= version.parse("1.6"):
    is_amp_available = True
    from torch.cuda.amp import autocast
else:
    is_amp_available = False

import transformers
from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput, 
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    Conv1D,
    PreTrainedModel,
    SequenceSummary,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from transformers import GPT2PreTrainedModel,GPT2LMHeadModel, AutoConfig
from transformers.models.gpt2.modeling_gpt2 import (
    PARALLELIZE_DOCSTRING, 
    DEPARALLELIZE_DOCSTRING, 
    GPT2_INPUTS_DOCSTRING,
    _CONFIG_FOR_DOC,
    GPT2Model
)

from modeling_bertwithoutpositionembedding import BertWithoutPositionEmbeddingModel, UncastableEmbedding

from transformers import BertConfig, GPT2Config


class CausalPragConfig(PretrainedConfig):

    def __init__(
        self,
        nuser=None,
        nitem=None,
        ui_hidden_size=None,
        bert_config=None,
        gpt_config=None,
        pretrained_gpt_path=None,
        average_rating=3.0,
        prag_prefix_length=1,
        num_latent_factor=5,
        use_user_topic=False,
        use_wide=False,
        **kwargs
    ):
        self.nuser = nuser
        self.nitem = nitem
        self.ui_hidden_size = ui_hidden_size
        self.gpt_config = gpt_config
        if pretrained_gpt_path is not None:
            self.gpt_config = AutoConfig.from_pretrained(pretrained_gpt_path)
        self.bert_config = bert_config
        self.pretrained_gpt_path = pretrained_gpt_path
        self.average_rating = average_rating
        self.prag_prefix_length = prag_prefix_length
        self.num_latent_factor = num_latent_factor
        self.use_user_topic = use_user_topic
        self.use_wide = use_wide
        
        # we use gpt2 tokens as the bert model is pooling over a knowledge source
        # hense from seq2seq perspective, gpt model should have the input
        if self.gpt_config is not None:
            super().__init__(bos_token_id=self.gpt_config.bos_token_id, eos_token_id=self.gpt_config.eos_token_id)
        else:
            super().__init__(**kwargs)
    
        
    def save_pretrained(self, save_directory, push_to_hub=False,**kwargs):
        self.bert_config.save_pretrained(os.path.join(save_directory, 'bert_config'))
        self.gpt_config.save_pretrained(os.path.join(save_directory, 'gpt_config'))
        # swap out bert/gpt config as they are not json-serializable
        tmp_bert_config = self.bert_config 
        tmp_gpt_config = self.gpt_config
        self.bert_config = None
        self.gpt_config = None
        super().save_pretrained(save_directory, **kwargs)
        self.bert_config = tmp_bert_config 
        self.gpt_config = tmp_gpt_config
        
    @classmethod 
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        bert_config = BertConfig.from_pretrained(os.path.join(pretrained_model_name_or_path, 'bert_config'))
        gpt_config = GPT2Config.from_pretrained(os.path.join(pretrained_model_name_or_path, 'gpt_config'))
        config_dict = cls._dict_from_json_file(os.path.join(pretrained_model_name_or_path, 'config.json'))
        config_dict['bert_config'] = bert_config
        config_dict['gpt_config'] = gpt_config
        config = CausalPragConfig(**config_dict)
        config.bert_config = bert_config 
        config.gpt_config = gpt_config
        return config, {} # HF except second param to be model kwargs, we turn this off for easy development

    def to_json_string(self, *args, **kwargs):
        tmp_bert_config = self.bert_config 
        tmp_gpt_config = self.gpt_config
        self.bert_config = None
        self.gpt_config = None
        if isinstance(tmp_bert_config, BertConfig):
            self.bert_config = 'transformers.BertConfig'
        if isinstance(tmp_gpt_config, GPT2Config):
            self.gpt_config = 'transformers.GPT2Config'
        out = super().to_json_string(*args, **kwargs)
        self.bert_config = tmp_bert_config 
        self.gpt_config = tmp_gpt_config
        return out

class CausalPragPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CausalPragConfig
    base_model_prefix = "transformer"
    is_parallelizable = False
    supports_gradient_checkpointing = False

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights. Copied from HF gpt2pretrainedmodel"""
        if isinstance(module, (nn.Linear, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        for name, p in module.named_parameters():
            if "c_proj" in name and "weight" in name:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(0.02 / math.sqrt(2 * self.config.gpt_config.n_layer)))


@dataclass
class CausalPragModelOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    regression_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    regression_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    evidence_loss: Optional[torch.FloatTensor] = None


class CausalPragNonlinearity(nn.Module):

    def __init__(self, embed_dim, config):
        super().__init__()
        self.c_fc = torch.nn.Linear(embed_dim, embed_dim)
        self.c_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.act = ACT2FN[config.gpt_config.activation_function]
        self.dropout = nn.Dropout(config.gpt_config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
    
class CausalPragRatingHead(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.user_bias = torch.nn.parameter.Parameter(
            torch.empty(config.nuser).normal_(mean=0,std=0.0001)
        )
        self.item_bias = torch.nn.parameter.Parameter(
            torch.empty(config.nitem).normal_(mean=0,std=0.0001)
        )

        self.user_factor = torch.nn.parameter.Parameter(
            torch.empty([config.nuser, config.ui_hidden_size]).normal_(mean=0,std=0.0001)
        )
        self.item_factor = torch.nn.parameter.Parameter(
            torch.empty([config.nitem, config.ui_hidden_size]).normal_(mean=0,std=0.0001)
        )
        # self.bn = torch.nn.BatchNorm1d(config.ui_hidden_size*2+config.bert_config.hidden_size)
        self.nonlinear_1 = CausalPragNonlinearity(config.ui_hidden_size*2+config.bert_config.hidden_size, config)
        self.nonlinear_2 = CausalPragNonlinearity(config.ui_hidden_size*2+config.bert_config.hidden_size, config)
        self.down_proj = torch.nn.Linear(config.ui_hidden_size*2+config.bert_config.hidden_size,1)
        self.alpha = config.average_rating

    def forward(self, userid, itemid, pooled_hidden_states, return_hidden=False):
        # print(userid.shape, itemid.shape, pooled_hidden_states.shape, self.user_factor[userid].shape)
        hidden = self.nonlinear_1(
            torch.cat([self.user_factor[userid],self.item_factor[itemid], pooled_hidden_states], dim=-1)
        )
        # hidden = self.bn(hidden)
        hidden = self.nonlinear_2(
            hidden
        )
        score = self.down_proj(hidden)
        score = self.alpha + self.user_bias[userid]  + self.item_bias[itemid] + score.squeeze(-1)
        
        if return_hidden:
            return score, hidden
        else:
            return score
    


class MLPAttentionPooling(torch.nn.Module):
    """
    self attention pooling
    """
    def __init__(self, input_dim, hidden_dim=32, do_pooling=False):
        super(MLPAttentionPooling, self).__init__()
        self.do_pooling = do_pooling
        self.attention_fc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )
    def forward(self, value, mask, **kwargs):
        attn_score = self.attention_fc(value)
        
        if self.do_pooling:
            attn_score = attn_score.masked_fill_(~mask.bool().view(attn_score.size()), -float('Inf'))
            attn_weights = torch.nn.functional.softmax(attn_score, dim=1)
            dot_prod = attn_weights.expand(attn_weights.shape[0],attn_weights.shape[1],value.shape[2]) * value
            wsum = torch.sum(dot_prod, dim=1)
            return wsum
        else:
            return attn_score

class MLPAttentionPoolingWithQuery(torch.nn.Module):
    """
    attention pooling with query
    """
    def __init__(self, input_dim, hidden_dim=32, do_pooling=False):
        super(MLPAttentionPoolingWithQuery, self).__init__()
        self.do_pooling = do_pooling
        self.attention_fc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )
    def forward(self, query, value, mask, **kwargs):
        attn_score = self.attention_fc(query)
        
        if self.do_pooling:
            attn_score = attn_score.masked_fill_(~mask.bool().view(attn_score.size()), -float('Inf'))
            attn_weights = torch.nn.functional.softmax(attn_score, dim=1)
            dot_prod = attn_weights.expand(attn_weights.shape[0],attn_weights.shape[1],value.shape[2]) * value
            wsum = torch.sum(dot_prod, dim=1)
            return wsum
        else:
            return attn_score



class CausalPragDecoderModel(CausalPragPreTrainedModel):
    
    def __init__(self, config):
        
        super().__init__(config)
        
        # embdding
        if config.bert_config.force_word_embeddings_on_cpu:
            self.word_embeddings =UncastableEmbedding(
                config.bert_config.vocab_size, config.bert_config.hidden_size, padding_idx=config.bert_config.pad_token_id
            )
        else:
            self.word_embeddings = nn.Embedding(config.bert_config.vocab_size, config.bert_config.hidden_size, padding_idx=config.bert_config.pad_token_id)
        
        for param in self.word_embeddings.parameters():
            param.requires_grad = False
            
            
        # gpt model
        if config.pretrained_gpt_path is None:
            self.gpt = GPT2LMHeadModel(config.gpt_config)
        else:
            print('loading gpt model...')
            # load the model
            gpt_config = AutoConfig.from_pretrained(config.pretrained_gpt_path)
            # override several parameters so we have a weak decoder
            # print('[lmhead warning: these config will be reset to lower value!]')
            # print('n_layer', gpt_config.n_layer, 'n_head', gpt_config.n_head, 'n_embd', gpt_config.n_embd)
            gpt_config.n_layer = 1
            gpt_config.n_head = 1
            # gpt_config.n_inner = 1024
            # set the config to be config of the prtrained model
            config.gpt_config = gpt_config
            # self.gpt = GPT2LMHeadModel(config.gpt_config)
            # init a gpt model
            self.gpt = GPT2LMHeadModel.from_pretrained(
                config.pretrained_gpt_path, 
                config=config.gpt_config,
                ignore_mismatched_sizes=True
            )
            # then we set it to None so 
            # we don't load the pretrained model again next time
            config.pretrained_gpt_path = None
        
        
        # set config
        self.config = config
        
        
            
        # project hidden of rating head to gpt embedding space
        self.lm_projection_heads = []
        for i in range(config.prag_prefix_length):
            lm_projection_head = torch.nn.Sequential(
                CausalPragNonlinearity(
                    embed_dim = config.bert_config.hidden_size, 
                    config = config
                ),
                torch.nn.Linear(
                    config.bert_config.hidden_size,
                    config.gpt_config.n_embd
                )
            )
            exec('self.lm_projection_head_'+str(i)+' = lm_projection_head')
            exec('self.lm_projection_heads.append(self.lm_projection_head_'+str(i)+')')
            
        self.prefix_length = config.prag_prefix_length
        

    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        review_history_input_ids=None,  
        review_history_token_type_ids=None,
        review_history_attention_mask=None,
        rating_labels=None,
        user=None,
        item=None,
        evidence_input_ids = None,
        evidence_attention_mask = None,
        retrieval_label = None,
        gold_review_id=None,
        return_embedding_only = False,
        rating_prediction_only = False,
        wsum=None
    ):
        
        
        if wsum == None:
            # get wsum
            wsum = self.word_embeddings(gold_review_id.cpu())
        
        # lm loss of the conditional version
        # get language model output
        # this logic was obtained from ClipClap (checkout their idea for piping CLIP and GPT together!)
        prefix = []
        for lm_head in self.lm_projection_heads:
            this_hidden = lm_head(wsum)
            prefix.append(this_hidden.unsqueeze(1))
        prefix = torch.cat(prefix, dim=1) # view prior as a token
        embedding_text = self.gpt.transformer.wte(input_ids)
        embedding_cat = torch.cat((prefix, embedding_text), dim=1) 
        if input_ids is not None:
            dummy_token = self.get_dummy_token(input_ids.shape[0], input_ids.device)
            lm_labels = torch.cat(
                (
                    dummy_token, 
                    input_ids.masked_fill(~attention_mask.bool(), -100)
                    ), dim=1)
        if attention_mask is not None:
            dummy_mask = torch.ones([input_ids.shape[0], self.prefix_length], device=input_ids.device)
            lm_masks = torch.cat((dummy_mask, attention_mask), dim=-1)
        lm_out = self.gpt(inputs_embeds=embedding_cat, labels=lm_labels, attention_mask=lm_masks)
        
        lm_loss = lm_out.loss
        loss = lm_loss
        
        return CausalPragModelOutput(
                loss = lm_loss,
                regression_loss = 0,
                evidence_loss = 0,
                lm_loss = lm_out.loss.item()
            )
    
    def get_dummy_token(self, batch_size: int, device):
        # return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)
        return torch.full((batch_size, self.prefix_length),-100).to(torch.int64).to(device)

    def inference(
        self, 
        wsum=None,
        beam_size = 5,
        temperature=1.0,
        tokenizer=None
    ):
        """
        beam searc decoding
        *not batched* 
        credit: clipcap (clip->gpt prompt)
        """
        
        prefix = []
        for lm_head in self.lm_projection_heads:
            this_hidden = lm_head(wsum)
            prefix.append(this_hidden.unsqueeze(1))
        prefix = torch.cat(prefix, dim=1) # view prior as a token
        embed = prefix
        
        batch_size = wsum.shape[0]
        
        dummy_token = self.get_dummy_token(batch_size, wsum.device)    
        dummy_mask = torch.ones([batch_size, self.prefix_length], device=wsum.device)
        
        model = self
        model.eval()
        stop_token_index = tokenizer.encode(tokenizer.eos_token)[0]
        tokens = None
        scores = None
        device = next(model.parameters()).device
        seq_lengths = torch.ones(beam_size, device=device)
        is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
        with torch.no_grad():
            
            generated = embed
            for i in range(30):
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                logits = logits.softmax(-1).log()
                if scores is None:
                    scores, next_tokens = logits.topk(beam_size, -1)
                    generated = generated.expand(beam_size, *generated.shape[1:])
                    next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                    if tokens is None:
                        tokens = next_tokens
                    else:
                        tokens = tokens.expand(beam_size, *tokens.shape[1:])
                        tokens = torch.cat((tokens, next_tokens), dim=1)
                else:
                    logits[is_stopped] = -float('Inf')
                    logits[is_stopped, 0] = 0
                    scores_sum = scores[:, None] + logits
                    seq_lengths[~is_stopped] += 1
                    scores_sum_average = scores_sum / seq_lengths[:, None]
                    scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                    next_tokens_source = next_tokens // scores_sum.shape[1]
                    seq_lengths = seq_lengths[next_tokens_source]
                    next_tokens = next_tokens % scores_sum.shape[1]
                    next_tokens = next_tokens.unsqueeze(1)
                    tokens = tokens[next_tokens_source]
                    tokens = torch.cat((tokens, next_tokens), dim=1)
                    generated = generated[next_tokens_source]
                    scores = scores_sum_average * seq_lengths
                    is_stopped = is_stopped[next_tokens_source]
                next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
                if is_stopped.all():
                    break
        scores = scores / seq_lengths
        output_list = tokens.cpu().numpy()
        output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
        order = scores.argsort(descending=True)
        output_texts = [output_texts[i] for i in order]
        return output_texts[0] # best result

        lm_out = self.gpt(inputs_embeds=embedding_cat, attention_mask=lm_masks)
        
        lm_loss = lm_out.loss
        loss = lm_loss
        
        return CausalPragModelOutput(
                loss = lm_loss,
                regression_loss = 0,
                evidence_loss = 0,
                lm_loss = lm_out.loss.item()
            )
    



if __name__ == '__main__':

    from transformers import BertModel, BertConfig
    bert_configuration = BertConfig(
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

    causal_prag_config = CausalPragConfig(
        nuser=50,
        nitem=25,
        ui_hidden_size=32,
        pretrained_gpt_path = 'distilgpt2',
        bert_config = bert_configuration
    )

    causal_prag = CausalPragModel(causal_prag_config) 