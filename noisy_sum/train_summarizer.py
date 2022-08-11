from datasets import load_dataset, Dataset
from transformers import  DataCollatorForSeq2Seq
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_DATASETS_CACHE"]=".cache/huggingface/datasets"
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
from transformers import EarlyStoppingCallback




def main(args):
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    dataset = load_dataset(
        'csv',
        data_files={
            'train': os.path.join(args.input_dir, 'train.csv'),
            'test': os.path.join(args.input_dir, 'test.csv'),
            'valid': os.path.join(args.input_dir, 'valid.csv')
            }
        )

    

    # load model, resize emb, etc.
    tokenizer = AutoTokenizer.from_pretrained("allenai/unifiedqa-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("allenai/unifiedqa-t5-base")
    special_tokens_dict = {'additional_special_tokens': ['<sep>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    def preprocess_function(examples):
        inputs = examples['source']
        targets = examples['target_review']
        model_inputs = tokenizer(inputs, max_length=512, padding='max_length', truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=20, padding='max_length', truncation=True)
            
        labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in labels['input_ids']] 
                ]

        model_inputs["labels"] = labels["input_ids"][0]
        
        return model_inputs

    train_dataset = dataset['train'].map(preprocess_function)#.set_format(type='torch')
    valid_dataset = dataset['valid'].map(preprocess_function)#.set_format(type='torch')

    print('train_dataset is ', train_dataset)


    trainer = Trainer(
        model=model,
        train_dataset=train_dataset, 
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        args=TrainingArguments(
            load_best_model_at_end = True,
            output_dir = args.output_dir,
            save_strategy = 'epoch',
            evaluation_strategy = 'epoch',
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=5e-5,
            num_train_epochs=20,
            save_total_limit =1,
            gradient_accumulation_steps = 4,
            logging_steps=128
        )
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience = 3,
        early_stopping_threshold=1e-7
    )
    trainer.add_callback(early_stopping_callback)
    trainer.train()


if __name__ == '__main__':
    # from easydict import EasyDict as edict
    import argparse
    parser = argparse.ArgumentParser(description='prepare data')
    parser.add_argument('--auto_arg_by_dataset', type=str, default=None,
                        help='auto argument')
    args = parser.parse_args()

    if args.auto_arg_by_dataset is not None:
        from easydict import EasyDict as edict
        assert args.auto_arg_by_dataset in ('yelp', 'movies_and_tv', 'tripadvisor')
        if args.auto_arg_by_dataset == 'tripadvisor':
            args = edict(
                dict(
                    input_dir='noisy_summarization_data/tripadvisor/',
                    output_dir='models/tripadvisor/'
                )
            )
        elif args.auto_arg_by_dataset == 'yelp':
            args = edict(
                dict(
                    input_dir='noisy_summarization_data/yelp/',
                    output_dir='models/yelp/'
                )
            )
        elif args.auto_arg_by_dataset == 'movies_and_tv':
            args = edict(
                dict(
                    input_dir='noisy_summarization_data/movies_and_tv/',
                    output_dir='models/movies_and_tv/'
                )
            )
    main(args)