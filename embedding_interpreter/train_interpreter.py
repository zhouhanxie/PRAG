import torch
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_DATASETS_CACHE"]=".cache/huggingface/datasets"
import numpy as np

import numpy as np
from transformers import Trainer
from utils import *
from huggingface_utils import *

from transformers import Trainer
import numpy as np

from modeling_prag_decoder import CausalPragConfig
from modeling_prag_decoder import CausalPragDecoderModel
from huggingface_utils import (
    prepare_dataset,
    CustomTrainer,
    CustomDataCollator,
    ProgressCallback,
    CustomDataCollator
)
from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import BertConfig
from transformers import TrainingArguments
from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback, TrainerCallback 

def main(args):
    print('preparing data...')
    train_dataset, valid_dataset, review_history = prepare_dataset(
        args.data_path,
        args.index_dir,
        maybe_load_from=args.cached_review_embedding
    )

    # fit vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', min_df=2)
    vectorizer.fit(train_dataset['text'])

    print('initializing prag model...')
    bert_configuration = BertConfig(
        vocab_size = review_history.nreview(),
        hidden_size = 768,
        num_hidden_layers = args.bert_num_hidden_layers,
        num_attention_heads = args.bert_num_attention_heads,
        intermediate_size = args.bert_intermediate_size,
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
        force_word_embeddings_on_cpu=args.force_document_embeddings_on_cpu,
        nuser=review_history.nuser(),
        nitem=review_history.nitem()
    )

    # model configuration
    # there are unused parameter from dev phase
    # need to clean this up in the future
    causal_prag_config = CausalPragConfig(
        nuser=review_history.nuser(),
        nitem=review_history.nitem(),
        ui_hidden_size=args.ui_hidden_size,
        pretrained_gpt_path = 'gpt2',
        bert_config = bert_configuration,
        average_rating = review_history.mean_rating(),
        prag_prefix_length=5,
        use_user_topic=True,
        use_wide=True
    )
    causal_prag = CausalPragDecoderModel(causal_prag_config) 
    
    print('tokenizing')
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token
    valid_dataset = valid_dataset.map(
            PreprocessFunction(
                tokenizer, 
                review_history,
                tfidf_vectorizer=vectorizer
                ),
            batched=False,
            desc="Running tokenizer on dataset"
    )

    train_dataset = train_dataset.map(
            PreprocessFunction(
                tokenizer, 
                review_history,
                tfidf_vectorizer=vectorizer
                ),
            batched=False,
            desc="Running tokenizer on dataset"
    )

    
    print('copying embedded document embeddings to prag model..')
    causal_prag.word_embeddings.weight.copy_(torch.from_numpy(review_history.text_table))
    
    print('initializing trainer...')
    trainer = CustomTrainer(
        model=causal_prag,
        train_dataset= train_dataset, #Dataset.from_pandas(train_dataset.to_pandas().head(128)),
        eval_dataset= valid_dataset, # Dataset.from_pandas(train_dataset.to_pandas().head(128)),
        tokenizer=tokenizer,
        data_collator=CustomDataCollator(
            tokenizer=tokenizer,
            input_ids_max_length=25
        ),
        args=TrainingArguments(
            load_best_model_at_end = True,
            output_dir = args.output_dir,
            save_strategy = 'epoch',
            evaluation_strategy = 'epoch',
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            save_total_limit = args.save_total_limit,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            logging_steps=128
        )
    )


    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience = args.tolerance_steps,
        early_stopping_threshold=1e-7
    )
    progress_callback = ProgressCallback().setup(total_epochs = args.epochs, print_every=500)
    trainer.add_callback(progress_callback)
    trainer.add_callback(early_stopping_callback)

    print('training...')
    trainer_output = trainer.train() 

    print('done')
    print(trainer_output)
    
    return trainer


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generator Model')
    parser.add_argument('--auto_arg_by_dataset', type=str, default=None,
                        help='auto argument')
    args = parser.parse_args()

    if args.auto_arg_by_dataset is not None:
        from easydict import EasyDict as edict
        assert args.auto_arg_by_dataset in ('yelp', 'movies_and_tv', 'tripadvisor')
        
        if args.auto_arg_by_dataset == 'tripadvisor':
            args = edict(
                dict(
                    bert_intermediate_size=1024, 
                    bert_num_attention_heads=1, 
                    bert_num_hidden_layers=2, 
                    data_path='../nlg4rec_data/TripAdvisor/reviews.pickle', 
                    epochs=100, 
                    force_document_embeddings_on_cpu=True, 
                    gradient_accumulation_steps=1, 
                    index_dir='../nlg4rec_data/TripAdvisor/1', 
                    log_interval=500, 
                    lr=5e-5, 
                    output_dir='./TripAdvisor/1', 
                    per_device_eval_batch_size=128, 
                    per_device_train_batch_size=128, 
                    save_total_limit=1, 
                    tolerance_steps=3, 
                    ui_hidden_size=768,
                    cached_review_embedding='transformer_retriever/tripadvisor_table.npy'
                )
            )
        elif args.auto_arg_by_dataset == 'yelp':
            args = edict(
                dict(
                    bert_intermediate_size=1024, 
                    bert_num_attention_heads=1, 
                    bert_num_hidden_layers=2, 
                    data_path='../nlg4rec_data/Yelp/reviews.pickle', 
                    epochs=100, 
                    force_document_embeddings_on_cpu=True, 
                    gradient_accumulation_steps=1, 
                    index_dir='../nlg4rec_data/Yelp/1', 
                    log_interval=500, 
                    lr=5e-5, 
                    output_dir='./Yelp/1', 
                    per_device_eval_batch_size=128, 
                    per_device_train_batch_size=128, 
                    save_total_limit=1, 
                    tolerance_steps=3, 
                    ui_hidden_size=768,
                    cached_review_embedding='archive/transformer_retriever/yelp_table.npy'
                )
            )
        elif args.auto_arg_by_dataset == 'movies_and_tv':
            args = edict(
                dict(
                    bert_intermediate_size=1024, 
                    bert_num_attention_heads=1, 
                    bert_num_hidden_layers=2, 
                    data_path='../nlg4rec_data/Amazon/MoviesAndTV/reviews.pickle', 
                    epochs=100, 
                    force_document_embeddings_on_cpu=True, 
                    gradient_accumulation_steps=1, 
                    index_dir='../nlg4rec_data/Amazon/MoviesAndTV/1', 
                    log_interval=500, 
                    lr=5e-5, 
                    output_dir='./Amazon/MoviesAndTV/1', 
                    per_device_eval_batch_size=128, 
                    per_device_train_batch_size=128, 
                    save_total_limit=1, 
                    tolerance_steps=3, 
                    ui_hidden_size=768,
                    cached_review_embedding='archive/transformer_retriever/amazon_moviesandtv_table.npy'
                )
            )
    
    main(args)