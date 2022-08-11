import torch
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_DATASETS_CACHE"]=".cache/huggingface/datasets"
import numpy as np
from modeling_personalized_retriever import PersonalizedRetrieverModel, PersonalizedRetrieverConfig
from transformers import BertConfig
from transformers import TrainingArguments
from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback, TrainerCallback 
from datasets import Dataset
from utils import DataReader
from tqdm import tqdm
from huggingface_utils import (
    prepare_dataset,
    CustomTrainer,
    CustomDataCollator,
    ProgressCallback,
    PreprocessFunction
)


def main(args):

    print('preparing data...')
    train_dataset, valid_dataset, review_history = prepare_dataset(
        args.data_path,
        args.index_dir,
        sentence_transformer_name = args.sentence_transformer_name, 
        maybe_load_from = args.cached_review_embedding
    )

    print('initializing retriever model...')
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

    retriever_config = PersonalizedRetrieverConfig(
        nuser=review_history.nuser(),
        nitem=review_history.nitem(),
        ui_hidden_size=args.ui_hidden_size,
        pretrained_gpt_path = 'distilgpt2',
        bert_config = bert_configuration,
        average_rating = review_history.mean_rating(),
        prag_prefix_length=6,
        num_latent_factor=args.num_latent_factor,
        use_user_topic=args.use_user_topic,
        use_wide=args.use_wide
    )

    retriever = PersonalizedRetrieverModel(retriever_config) 

    print('copying embedded word embeddings to prag model..')
    retriever.set_document_embedding(torch.from_numpy(review_history.text_table))
    

    # legacy code, actually the retriever don't need tokenized input_ids
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print('tokenizing')

    train_dataset = train_dataset.map(
            PreprocessFunction(
                tokenizer, 
                review_history
                ),
            batched=False,
            desc="Running tokenizer on dataset"
    )
    valid_dataset = valid_dataset.map(
            PreprocessFunction(
                tokenizer, 
                review_history
                ),
            batched=False,
            desc="Running tokenizer on dataset"
    )


    print('initializing trainer...')    
    trainer = CustomTrainer(
        model=retriever,
        train_dataset=train_dataset, # Dataset.from_pandas(train_dataset.to_pandas().head(256)),
        eval_dataset=valid_dataset, # Dataset.from_pandas(valid_dataset.to_pandas().head(256)),
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
            gradient_accumulation_steps = args.gradient_accumulation_steps
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

    return trainer.model, review_history


def eval_on_test(model, review_history, args):

    model = model.to(torch.device('cuda'))
    corpus = DataReader(args.data_path, args.index_dir)
    
    def prepare_model_input(userid, itemid, device=torch.device('cuda')):
    
        # print('here')
        user = torch.tensor([userid])
        item = torch.tensor([itemid])
        review_history_input_ids = []
        review_history_token_type_ids = []
        review_history_input_ids += review_history.get_user_history(
            user=userid, hide_item=itemid, return_embedding=False
        )
        while len(review_history_token_type_ids) < len(review_history_input_ids):
            review_history_token_type_ids.append(0)
        review_history_input_ids += review_history.get_item_history(
            item=itemid, hide_user=userid, return_embedding=False
        )

        # print('here')
        while len(review_history_token_type_ids) < len(review_history_input_ids):
            review_history_token_type_ids.append(1)

        output = (
            torch.atleast_1d(user).to(device), 
            torch.atleast_1d(item).to(device),
            torch.atleast_2d(torch.tensor(review_history_input_ids)).to(device), 
            torch.atleast_2d(torch.tensor(review_history_token_type_ids)).to(device), 
            torch.atleast_2d(torch.ones(len(review_history_input_ids)).int()).to(device)
        )

        return output
    
    ### loop over test samples
    r_preds = []
    with torch.no_grad():
        for sample in tqdm(corpus.test):
            user, item, review_history_input_ids, review_history_token_type_ids, review_history_attention_mask = \
            prepare_model_input(sample['user'], sample['item'], device=torch.device('cuda'))
            r_pred = model(
                user = user, 
                item = item, 
                review_history_input_ids = review_history_input_ids, 
                review_history_token_type_ids = review_history_token_type_ids, 
                review_history_attention_mask = review_history_attention_mask,
                rating_labels = torch.atleast_2d(torch.tensor(sample['rating'])).to(torch.device('cuda')),
                rating_prediction_only = True
            ).detach().item()
            if r_pred > corpus.max_rating:
                r_pred = corpus.max_rating 
            elif r_pred < corpus.min_rating:
                r_pred = corpus.min_rating 
            else:
                pass
            r_preds.append(r_pred)
            
    from sklearn.metrics import mean_squared_error

    rmse = mean_squared_error(
        y_true = [sample['rating'] for sample in corpus.test],
        y_pred = r_preds,
        squared = False
    )

    print('TEST RMSE: ', rmse)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Personalized Retriever model')
    parser.add_argument('--data_path', type=str, default=None,
                        help='path for loading the pickle data')
    parser.add_argument('--index_dir', type=str, default=None,
                        help='load indexes')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='upper epoch limit')
    parser.add_argument('--save_total_limit', type=int, default=1,
                        help='total limit of models to save')
    parser.add_argument('--per_device_train_batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='grad acc')
    parser.add_argument('--log_interval', type=int, default=500,
                        help='report interval')
    parser.add_argument('--output_dir', type=str, default='./tmp_output',
                        help='place to save essentially everying, args, weight, etc.')
    parser.add_argument('--tolerance_steps', type=int, default=3,
                        help='stop if consequtive n epochs brings no improvement')
    parser.add_argument('--ui_hidden_size', type=int, default=32,
                        help='hidden representation of user/item')
    parser.add_argument('--num_latent_factor', type=int, default=5,
                        help='latent of user/item')
    parser.add_argument('--bert_intermediate_size', type=int, default=768,
                        help='bert model intermediate size')
    parser.add_argument('--bert_num_hidden_layers', type=int, default=1,
                        help='bert model number of hidden layers')
    parser.add_argument('--bert_num_attention_heads', type=int, default=1,
                        help='bert model number of attention heads')
    parser.add_argument('--force_document_embeddings_on_cpu', action='store_true',
                        help='keep document embeddings on cpu')
    parser.add_argument('--use_user_topic', action='store_true',
                        help='semantic meaning is drawn from user')
    parser.add_argument('--use_wide', action='store_true',
                        help='use an additional wide component over full semantic feature')
    parser.add_argument('--cached_review_embedding', type=str, default=None,
                        help='place to try to load/save cached_review_embedding')
    parser.add_argument('--sentence_transformer_name', type=str, default='all-mpnet-base-v2',
                        help='sentence transformer for embedding reviews')
    
    args = parser.parse_args()
    print(args)
    best_model, review_history = main(args)
    eval_on_test(best_model, review_history, args)


    


    

    