import pandas as pd
from datasets import Dataset
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_DATASETS_CACHE"]="./huggingface_cache"
from transformers import T5Tokenizer, T5ForConditionalGeneration
from huggingface_utils import CustomTrainer, CustomDataCollator, Trainer, CustomTrainingArguments
from transformers import TrainingArguments
from transformers import  DataCollatorForSeq2Seq

def assemble_model_input(prompt, topic, evidence):
    """
    util for packing model input together.
    prompt:
        a question, potentially ended with new line character
    topic:
        list of noisy topic terms
    evidence:
        a string of evidence to use for the reader
    """
    out = prompt.replace('\\n','')
    out += '['+', '.join(topic)+']'
    out += '\\n'
    out += evidence
    out = out.lower()
    return out

def to_bitfit(model, verbose=False):
    """
    turn off anything except bias and classification head 
    in a transformer model
    """
    if verbose:
        print('most parameters will be turned off.')
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.lm_head.named_parameters():
        param.requires_grad = True
        
    return model

def main(args):
    data = pd.read_csv(args.data_path)
    source = data.apply(lambda row: assemble_model_input(row['prompt'], eval(row['topic']), row['evidence']), axis=1).tolist()
    target = data['corrected_summary'].tolist()
    train_df = pd.DataFrame({'source':source, 'target':target})
    train_dataset = Dataset.from_pandas(train_df)

    tokenizer = T5Tokenizer.from_pretrained(args.base_model_name_or_path)

    def preprocess_function(examples):
        inputs = examples['source']
        targets = examples['target']
        model_inputs = tokenizer(inputs, max_length=50, padding='max_length', truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=50, padding='max_length', truncation=True)
            
        labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in labels['input_ids']] 
                ]

        model_inputs["labels"] = labels["input_ids"][0] # [0] for 2d->1d
        
        return model_inputs

    train_dataset = train_dataset.map(preprocess_function)
    model = T5ForConditionalGeneration.from_pretrained(args.base_model_name_or_path)

    # model = to_bitfit(model, verbose=True)

    data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
        )

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_dataset, 
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=CustomTrainingArguments(
            load_best_model_at_end = False,
            output_dir = args.output_dir,
            save_strategy = 'epoch',
            per_device_train_batch_size=args.per_device_train_batch_size,
            learning_rate=3e-5,
            num_train_epochs=args.num_train_epochs,
            save_total_limit =1,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            logging_steps=1
        )
    )

    trainer.train()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generator Model')
    parser.add_argument('--auto_arg_by_dataset', type=str, default=None,
                        help='auto argument')
    args = parser.parse_args()

    if args.auto_arg_by_dataset is not None:
        from easydict import EasyDict as edict
        assert args.auto_arg_by_dataset in ('yelp', 'movies_and_tv', 'tripadvisor')
        if args.auto_arg_by_dataset == 'movies_and_tv':
            args = edict(
                dict(
                    data_path='./annotated_data/movies_and_tv_annotated.csv', 
                    base_model_name_or_path="allenai/unifiedqa-t5-3b",
                    output_dir='./movies_and_tv/',
                    num_train_epochs=10,
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=12
                )
            )
        elif args.auto_arg_by_dataset == 'yelp':
            args = edict(
                dict(
                    data_path='./annotated_data/yelp_annotated.csv', 
                    base_model_name_or_path="allenai/unifiedqa-t5-3b",
                    output_dir='./yelp/',
                    num_train_epochs=10,
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=12
                )
            )
        elif args.auto_arg_by_dataset == 'tripadvisor':
            args = edict(
                dict(
                    data_path='./annotated_data/tripadvisor_annotated.csv', 
                    base_model_name_or_path="allenai/unifiedqa-t5-3b",
                    output_dir='./tripadvisor/',
                    num_train_epochs=10,
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=12
                )
            )
    
    main(args)
