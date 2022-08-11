from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import sys
sys.path.insert(0, '../personalized_retriever')
from huggingface_utils import move_to_device
import pandas as pd
from tqdm import tqdm

device = torch.device('cuda')




def main(args):

    qa_model = AutoModelForSeq2SeqLM.from_pretrained('allenai/unifiedqa-t5-base')
    tokenizer = AutoTokenizer.from_pretrained(
        'allenai/unifiedqa-t5-base'
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.summarizer_path
    )
    
    model = model.to(device)
    qa_model = qa_model.to(device)
    
    dat_in = pd.read_csv(args.input_csv)
    dat = dat_in['evidence'].tolist()
    dat = [eval(i) for i in dat]
    dat = ['<sep>'.join(i) for i in dat]

    def run_model(model_ptr, input_string, **generator_args):
        input_ids = tokenizer.encode(input_string, return_tensors="pt")
        input_ids = move_to_device(input_ids, device)
        res = model_ptr.generate(input_ids, **generator_args)
        return tokenizer.batch_decode(res, skip_special_tokens=True)
    
    all_out = []
    for datum in tqdm(dat, disable=False):
        summary = run_model(model, datum, num_beams=1)[0]
        justification = run_model(qa_model, 'why '+summary+' ?\n'+datum.replace('<sep>',' '), num_beams=1)[0]
        out = {'summary':summary, 'justification':justification}
        # print(out)
        all_out.append(out)
        
    dat_in['summarizer_generated'] = all_out
    
    dat_in.to_csv(args.output_csv)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Personalized Retriever model inference')
    parser.add_argument('--auto_arg_by_dataset', type=str, default=None,
                        help='auto argument')
    args = parser.parse_args()

    if args.auto_arg_by_dataset is not None:
        from easydict import EasyDict as edict
        assert args.auto_arg_by_dataset in ('yelp', 'movies_and_tv', 'tripadvisor')
        if args.auto_arg_by_dataset == 'tripadvisor':
            args = edict(
                dict(
                    input_csv='prag_generation/generated/tripadvisor_generated.csv',
                    summarizer_path='noisy_sum/models/tripadvisor/checkpoint-2634',
                    output_csv='./tripadvisor_generated.csv'
                )
            )
        elif args.auto_arg_by_dataset == 'yelp':
            args = edict(
                dict(
                    input_csv='prag_generation/generated/yelp_generated.csv',
                    summarizer_path='prank/noisy_sum/models/yelp/checkpoint-10956',
                    output_csv='./yelp_generated.csv'
                )
            )
        elif args.auto_arg_by_dataset == 'movies_and_tv':
            args = edict(
                dict(
                    input_csv='prag_generation/generated/movies_and_tv_generated.csv',
                    summarizer_path='prank/noisy_sum/models/movies_and_tv/checkpoint-7908',
                    output_csv='./movies_and_tv_generated.csv'
                )
            )
    main(args)