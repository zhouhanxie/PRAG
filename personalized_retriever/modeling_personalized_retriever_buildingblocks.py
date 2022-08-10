"""Useful Building Blocks for Personalized Retriever Model"""
from collections import OrderedDict
from typing import Any, List, Mapping, Optional

from transformers import PreTrainedTokenizer, TensorType, is_torch_available

from transformers.configuration_utils import PretrainedConfig
from transformers.onnx import OnnxConfigWithPast, PatchingSpec
from transformers.utils import logging

logger = logging.get_logger(__name__)

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

from modeling_bertwithoutpositionembedding import BertWithoutPositionEmbeddingModel

from transformers import BertConfig, GPT2Config

@dataclass
class PersonalizedRetrieverModelOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    regression_loss: Optional[torch.FloatTensor] = None


class PersonalizedRetrieverNonlinearity(nn.Module):

    def __init__(self, embed_dim, config):
        super().__init__()
        self.c_fc = torch.nn.Linear(embed_dim, embed_dim)
        self.c_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.act = ACT2FN['gelu_new']
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class MLPSelfAttentionPooling(torch.nn.Module):
    """
    self attention pooling with a non-linear attn score predictor
    """
    def __init__(self, input_dim, hidden_dim=32, do_pooling=False):
        super(MLPSelfAttentionPooling, self).__init__()
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


class UIEmbeddingForRecommendation(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.user_factor = torch.nn.parameter.Parameter(
            torch.empty([config.nuser, config.ui_hidden_size]).normal_(mean=0,std=0.0001)
        )
        self.item_factor = torch.nn.parameter.Parameter(
            torch.empty([config.nitem, config.ui_hidden_size]).normal_(mean=0,std=0.0001)
        )

    def get_user_embedding(self, user):
        return self.user_factor[user]

    def get_item_embedding(self, item):
        return self.item_factor[item]

class PersonalizedRetrieverConfig(PretrainedConfig):

    def __init__(
        self,
        nuser=None,
        nitem=None,
        ui_hidden_size=None,
        bert_config=None,
        average_rating=3.0,
        num_latent_factor=5,
        use_user_topic=False,
        use_wide=False,
        **kwargs
    ):
        self.nuser = nuser
        self.nitem = nitem
        self.ui_hidden_size = ui_hidden_size
        self.bert_config = bert_config
        self.average_rating = average_rating
        self.num_latent_factor = num_latent_factor
        self.use_user_topic = use_user_topic
        self.use_wide = use_wide
        
        super().__init__(**kwargs)
        
    def save_pretrained(self, save_directory, push_to_hub=False,**kwargs):
        self.bert_config.save_pretrained(os.path.join(save_directory, 'bert_config'))
        tmp_bert_config = self.bert_config 
        self.bert_config = None
        super().save_pretrained(save_directory, **kwargs)
        self.bert_config = tmp_bert_config 
        
    @classmethod 
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        bert_config = BertConfig.from_pretrained(os.path.join(pretrained_model_name_or_path, 'bert_config'))
        config_dict = cls._dict_from_json_file(os.path.join(pretrained_model_name_or_path, 'config.json'))
        config_dict['bert_config'] = bert_config
        config = PersonalizedRetrieverConfig(**config_dict)
        config.bert_config = bert_config 
        return config, {} # HF except second param to be model kwargs, we just turn this off for 

    def to_json_string(self, *args, **kwargs):
        tmp_bert_config = self.bert_config 
        self.bert_config = None
        if isinstance(tmp_bert_config, BertConfig):
            self.bert_config = 'transformers.BertConfig'
        out = super().to_json_string(*args, **kwargs)
        self.bert_config = tmp_bert_config 
        return out


def combine_losses_with_potential_none(*args):
    """
    Given a bunch of losses possibly containing none
       return the sum of these terms
    Return None of all of the losses are none
    """
    out = None 
    for arg in args:
        if arg is not None:
            if out is None:
                out = arg 
            else:
                out += arg 
    return out

class PersonalizedRetrieverPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = PersonalizedRetrieverConfig
    base_model_prefix = "transformer"
    is_parallelizable = False
    supports_gradient_checkpointing = False

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
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
