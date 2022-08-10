"""
Had to override a series of HF classes for training with custom model & losses to work. 
Dumping them here for readability.
"""
import os
import math
import torch
import pandas as pd
import numpy as np
from copy import deepcopy

from dataclasses import dataclass
from transformers import BertModel, BertConfig
from transformers import PreTrainedTokenizerBase
from transformers import AutoTokenizer
from transformers import Trainer
from transformers.trainer_utils import PredictionOutput,EvalLoopOutput
from typing import Optional, Union
from datasets import Dataset
from copy import deepcopy
from sentence_transformers import SentenceTransformer

from transformers import EarlyStoppingCallback, TrainerCallback 
import time
import datetime

from utils import (
    ReviewHistory, 
    DataReader,
    move_to_device
)

from sklearn.metrics import balanced_accuracy_score

class ProgressCallback(TrainerCallback):

    def setup(self, total_epochs, print_every=1): 
        self.total_epochs = total_epochs 
        self.current_epoch = 0
        self.epoch_start_time = None
        self.current_step = 1
        self.global_start_time = time.time()
        self.print_every=print_every
        return self

    def on_step_begin(self, args, state, control, **kwargs):
        
        avg_time_per_step = (time.time() - self.global_start_time)/max(state.global_step,1 )
        eta = avg_time_per_step * (state.max_steps-state.global_step) / 3600
        if self.current_step % self.print_every == 0:
            print(
                'epoch: ', 
                self.current_epoch,
                ', step ',
                self.current_step, 
                '/', 
                state.max_steps // self.total_epochs, 
                '||', 
                datetime.datetime.now(),
                '|| ETA(hrs): ',
                round(eta,2)
                )
        self.current_step += 1
        

    def on_epoch_begin(self, args, state, control, **kwargs):
        print('[ProgressCallback]: current epoch: ', self.current_epoch,' / ', self.total_epochs)
        self.current_epoch += 1
        self.current_step = 1
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        print('[ProgressCallback]: epoch', self.current_epoch,' / ', self.total_epochs, ' done')
        print("--- %s hours ---" % ((time.time() - self.epoch_start_time)/3600) )

class PreprocessFunction:

    def __init__(self, tokenizer, review_history, tfidf_vectorizer, k=5):
        self.tokenizer = tokenizer 
        self.review_history = review_history
        self.tfidf_vectorizer = tfidf_vectorizer
        self.feature_names = np.array(self.tfidf_vectorizer.get_feature_names())
        self.k = k
        
    def _get_topk_keywords(self, example_text, k=5):
        eg = self.tfidf_vectorizer.transform([example_text])
        out = ', '.join(self.feature_names[np.asarray(np.argsort(-eg.todense())).squeeze()[:5]])
        return out
        
    def __call__(self, example):

        # Tokenize the texts
        result = self.tokenizer(self._get_topk_keywords(example['text'], self.k), 
                                padding=False, 
                                max_length=20, 
                                truncation=True
                               )
        # add prefix and post fix
        for key, value in self.tokenizer('generated#').items():
            result[key] = value+result[key]
        for key, value in self.tokenizer(self.tokenizer.eos_token).items():
            result[key] = result[key]+value  
        
        # done
        review_history_input_ids = []
        review_history_token_type_ids = []
        review_history_input_ids += self.review_history.get_user_history(
            user=example['user'], hide_item=example['item'], return_embedding=False
        )
        while len(review_history_token_type_ids) < len(review_history_input_ids):
            review_history_token_type_ids.append(0)
        review_history_input_ids += self.review_history.get_item_history(
            item=example['item'], hide_user=example['user'], return_embedding=False
        )
        while len(review_history_token_type_ids) < len(review_history_input_ids):
            review_history_token_type_ids.append(1)

        example['gold_review_id'] = self.review_history.get_ui_review(example['user'], example['item'], return_embedding=False)
            
        result['review_history_token_type_ids'] = review_history_token_type_ids
        result['review_history_input_ids'] = review_history_input_ids
        result['review_history_attention_mask'] = torch.ones(len(review_history_input_ids)).int().tolist()
        result['rating_labels'] = example['rating'] 
        result['labels'] = deepcopy(result['input_ids'])
        
                
        return result


@dataclass
class CustomDataCollator:

    tokenizer: PreTrainedTokenizerBase
    padding: bool = True
    input_ids_max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    history_pad_token_id: int = 0

    def __call__(self, features):
        review_history_input_ids = [feature["review_history_input_ids"] for feature in features] \
        if "review_history_input_ids" in features[0].keys() else None
        
        evidence_input_ids = [feature["evidence_input_ids"] for feature in features] \
        if "evidence_input_ids" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.

        # same for evidence ids, if they are present.
        if review_history_input_ids is not None:
            max_history_length = max(len(l) for l in review_history_input_ids)
        if evidence_input_ids is not None:
            max_evidence_length = max(len(l) for l in evidence_input_ids)
        
        # get padding side
        padding_side = self.tokenizer.padding_side

        # now we pad in a for loop
        for feature in features:
            if review_history_input_ids is not None:
                remainder = [self.history_pad_token_id] * (max_history_length - len(feature["review_history_input_ids"]))
                feature["review_history_input_ids"] = (
                    feature["review_history_input_ids"] + remainder if padding_side == "right" else remainder + feature["review_history_input_ids"]
                )
                feature["review_history_token_type_ids"] = (
                    feature["review_history_token_type_ids"] + remainder if padding_side == "right" else remainder + feature["review_history_token_type_ids"]
                )
                feature["review_history_attention_mask"] = (
                    feature["review_history_attention_mask"] + remainder if padding_side == "right" else remainder + feature["review_history_attention_mask"]
                )

                if 'retrieval_label' in feature:
                    feature["retrieval_label"] = (
                        feature["retrieval_label"] + remainder if padding_side == "right" else remainder + feature["retrieval_label"]
                    )

            if evidence_input_ids is not None:
                evidence_remainder = [self.tokenizer.pad_token_id] * (max_evidence_length - len(feature["evidence_input_ids"]))
                if 'evidence_input_ids' in feature:
                    feature["evidence_input_ids"] = (
                        feature["evidence_input_ids"] + evidence_remainder if padding_side == "right" else evidence_remainder + feature["evidence_input_ids"]
                    )

                if 'evidence_attention_mask' in feature:
                    feature["evidence_attention_mask"] = (
                        feature["evidence_attention_mask"] + evidence_remainder if padding_side == "right" else evidence_remainder + feature["evidence_attention_mask"]
                    )


        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.input_ids_max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )


        return features
    
class CustomTrainer(Trainer):
    
    def prediction_step(self,model,inputs,prediction_loss_only, ignore_keys):
        with torch.no_grad():
            if torch.cuda.is_available():
                output = model(
                    **move_to_device(inputs, torch.device('cuda'))
                )
            else:
                output = model(**inputs)
        
        return (output.loss.detach().cpu(), output.regression_loss, output.evidence_loss, output.lm_loss)

    def evaluation_loop(
            self,
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        ):
            """
            Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
            Works both with or without labels.
            """
            args = self.args
            model = self._wrap_model(self.model, training=False)
            self.callback_handler.eval_dataloader = dataloader
            model.eval()
            batch_size = dataloader.batch_size
            num_examples = self.num_examples(dataloader)
            print(f"***** Running evaluation loop *****")
            print(f"  Num examples = {num_examples}")
            print(f"  Batch size = {batch_size}")
            loss_host = []
            regression_loss_host= []
            evidence_loss_host= []
            lm_loss_host= []
            
            for step, inputs in enumerate(dataloader):
                loss, regression_loss, evidence_loss, lm_loss = self.prediction_step(
                    model, 
                    inputs, 
                    prediction_loss_only, 
                    ignore_keys=ignore_keys
                )
                
                loss_host += loss.repeat(batch_size).tolist()
                regression_loss_host.append(regression_loss)
                evidence_loss_host.append(evidence_loss)
                lm_loss_host.append(lm_loss)
                
                # call progress bar update, etc.
                self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)
            
            loss_host = torch.tensor(loss_host)
            
            
                    
            metrics = {
                'eval_loss':torch.mean(loss_host).item(),
                'lm_loss': np.mean(lm_loss_host),
            }

            print('[CustomTrainer]: Evaluation done)', metrics)

            output = EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=num_examples)
            return output





def prepare_dataset(data_path, index_dir, sentence_transformer_name='all-mpnet-base-v2', maybe_load_from=None):
    corpus = DataReader(
        data_path,
        index_dir
    )

    review_history = ReviewHistory(
        corpus.train, valid_data = corpus.valid, test_data = corpus.test
    ).build_embedded_text_table(
        SentenceTransformer(sentence_transformer_name), 
        torch.device('cuda'),
        maybe_load_from=maybe_load_from
    )

    # from prepare_evidence import prepare_evidence
    # prepare_evidence(corpus, review_history)

    train_dataframe = pd.DataFrame(corpus.train)
    valid_dataframe = pd.DataFrame(corpus.valid)

    train_dataset = Dataset.from_pandas(train_dataframe)
    valid_dataset = Dataset.from_pandas(valid_dataframe)
    
    
    return train_dataset, valid_dataset, review_history