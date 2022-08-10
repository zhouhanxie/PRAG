"""Personalized Retriever Model"""
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import transformers
from transformers.utils import logging
logger = logging.get_logger(__name__)

from modeling_bertwithoutpositionembedding import BertWithoutPositionEmbeddingModel
from modeling_personalized_retriever_buildingblocks import (
    PersonalizedRetrieverPreTrainedModel,
    PersonalizedRetrieverModelOutput,
    PersonalizedRetrieverNonlinearity,
    MLPSelfAttentionPooling,
    UIEmbeddingForRecommendation,
    combine_losses_with_potential_none,
    PersonalizedRetrieverConfig
)


class NHFTHead(torch.nn.Module):
    """
    a hidden factor head as described in Paper
        "Understanding Rating Dimensions with Review Text",
    with a optional wide component Inspired by Paper
        "Wide & Deep Learning for Recommender Systems"
    """
    def __init__(self, config):
        super(NHFTHead, self).__init__()
        self.config = config
        self.rating_downproj = torch.nn.Sequential(
            torch.nn.Linear(config.bert_config.hidden_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 12),
            torch.nn.ReLU(),
            torch.nn.Linear(12, self.config.num_latent_factor)
        )
        self.user_bias = torch.nn.parameter.Parameter(
            torch.empty(config.nuser).normal_(mean=0,std=0.0001)
        )
        self.item_bias = torch.nn.parameter.Parameter(
            torch.empty(config.nitem).normal_(mean=0,std=0.0001)
        )
        if not config.use_user_topic:
            self.user_factor = torch.nn.parameter.Parameter(
                torch.empty([config.nuser, self.config.num_latent_factor]).normal_(mean=0,std=0.0001)
            )
            self.rec = lambda userid,itemid,hid : self.user_bias[userid] + self.item_bias[itemid] + self.config.average_rating\
            +(self.rating_downproj(hid) * self.user_factor[userid]).sum(dim=-1)

        else:
            self.item_factor = torch.nn.parameter.Parameter(
                torch.empty([config.nitem, self.config.num_latent_factor]).normal_(mean=0,std=0.0001)
            )
            self.rec = lambda userid,itemid,hid : self.user_bias[userid] + self.item_bias[itemid] + self.config.average_rating\
            +(self.rating_downproj(hid) * self.item_factor[itemid]).sum(dim=-1)

        # wide component 
        if self.config.use_wide:
            print('[PersonalizedRetriever]: adding an extra wide component for HFT model')
            self.wide = torch.nn.Sequential(
                torch.nn.Linear(config.bert_config.hidden_size, config.bert_config.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(config.bert_config.hidden_size, 1)
            )
            tmp_lambda =  self.rec
            self.rec = lambda userid,itemid,hid : tmp_lambda(userid, itemid, hid) + self.wide(hid).squeeze(-1)

    def forward(self, user, item, topic_representation):
        rating_pred = self.rec(user,item, topic_representation)
        return rating_pred

class PersonalizedRetrieverModel(PersonalizedRetrieverPreTrainedModel):
    
    def __init__(self, config):
        
        super().__init__(config)

        # set config
        self.config = config
        # we'll have to train this bert-based pooler from scratch
        self.bert = BertWithoutPositionEmbeddingModel(config.bert_config)
        # personalized attention
        self.personalized_attention = MLPSelfAttentionPooling(config.ui_hidden_size*2+config.bert_config.hidden_size, do_pooling=True)
        # down projection
        self.down_proj = torch.nn.Sequential(
            torch.nn.Linear(config.ui_hidden_size*2+config.bert_config.hidden_size, config.bert_config.hidden_size),
            torch.nn.ReLU(),
            PersonalizedRetrieverNonlinearity(config.bert_config.hidden_size, config),
            torch.nn.ReLU(),
            PersonalizedRetrieverNonlinearity(config.bert_config.hidden_size, config),
            torch.nn.ReLU(),
            PersonalizedRetrieverNonlinearity(config.bert_config.hidden_size, config)
        )
        
        # for recommendation
        self.ui_embedding_table = UIEmbeddingForRecommendation(config)
        self.rating_regression_head = NHFTHead(config) 
        

    def set_document_embedding(self, data):
        self.bert.embeddings.word_embeddings.weight.copy_(data)
        return self
        

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
        rating_prediction_only = False
    ):
        
        # hook user and item representations into buffer
        # this will be added to word embeddings during embedding lookup
        self.bert.embeddings.additive_embeddings.append(
            self.ui_embedding_table.get_user_embedding(user).unsqueeze(-2).repeat(1, review_history_input_ids.shape[1], 1)
        )
        self.bert.embeddings.additive_embeddings.append(
            self.ui_embedding_table.get_item_embedding(item).unsqueeze(-2).repeat(1, review_history_input_ids.shape[1], 1)
        )

        # compute hidden document representation
        bert_hidden, bert_mask = self.bert.get_hidden_states_and_attention(
            review_history_input_ids, 
            review_history_token_type_ids,
            review_history_attention_mask
        ) 

        # pass through a MLP for retrieval prediction
        # yields a weighted sum of historical document (transformed)
        wsum = self.personalized_attention(
                value = torch.cat([
                    bert_hidden, 
                    self.ui_embedding_table.user_factor[user].unsqueeze(-2).repeat(1, bert_hidden.shape[1], 1), 
                    self.ui_embedding_table.item_factor[item].unsqueeze(-2).repeat(1, bert_hidden.shape[1], 1)
                    ], dim=-1), 
                mask = bert_mask
                )
        wsum = self.down_proj(wsum) 
        
        # inference time utils
        if rating_prediction_only:
            rating_pred = self.rating_regression_head.rec(user,item, wsum)
            return rating_pred

        # compute loss
        if rating_labels is not None:
            rating_pred = self.rating_regression_head.rec(user,item, wsum)
            rating_loss = torch.nn.functional.mse_loss(
                input = rating_pred, 
                target = rating_labels.float(),
                reduction='mean'
            )
        
        if gold_review_id is not None:
            retrieval_loss = torch.nn.functional.mse_loss(
                input = wsum, 
                target = self.bert.embeddings.word_embeddings(gold_review_id.cpu()),
                reduction='none'
            )
            retrieval_loss = torch.mean(retrieval_loss.sum(dim=-1))
        
        loss = combine_losses_with_potential_none(retrieval_loss, rating_loss)
        
        
        return PersonalizedRetrieverModelOutput(
                loss = loss,
                regression_loss = rating_loss.item(),
                lm_loss = retrieval_loss.item()
            )

    def predict_review_embedding(
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
        rating_prediction_only = False
    ):
        with torch.no_grad():
            # hook user and item representations into buffer
            # this will be added to word embeddings during embedding lookup
            self.bert.embeddings.additive_embeddings.append(
                self.ui_embedding_table.get_user_embedding(user).unsqueeze(-2).repeat(1, review_history_input_ids.shape[1], 1)
            )
            self.bert.embeddings.additive_embeddings.append(
                self.ui_embedding_table.get_item_embedding(item).unsqueeze(-2).repeat(1, review_history_input_ids.shape[1], 1)
            )

            # compute hidden document representation
            bert_hidden, bert_mask = self.bert.get_hidden_states_and_attention(
                review_history_input_ids, 
                review_history_token_type_ids,
                review_history_attention_mask
            ) 

            # pass through a MLP for retrieval prediction
            # yields a weighted sum of historical document (transformed)
            wsum = self.personalized_attention(
                    value = torch.cat([
                        bert_hidden, 
                        self.ui_embedding_table.user_factor[user].unsqueeze(-2).repeat(1, bert_hidden.shape[1], 1), 
                        self.ui_embedding_table.item_factor[item].unsqueeze(-2).repeat(1, bert_hidden.shape[1], 1)
                        ], dim=-1), 
                    mask = bert_mask
                    )
            wsum = self.down_proj(wsum) 
        
        return wsum

    def predict_rating_with_embedding(
        self, 
        embedding=None
    ):
        with torch.no_grad():
            rating_pred = self.rating_regression_head.rec(user,item, embedding)
        return rating_pred


if __name__ == '__main__':
    pass