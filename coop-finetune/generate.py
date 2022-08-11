import json
from typing import List
import pandas as pd
import torch
import rouge
from coop import VAE, util

from tqdm import tqdm
import pandas as pd

def main(args):
    vae = VAE('megagonlabs/optimus-yelp')

    vae.model.load_state_dict(
        torch.load(
            args.optimus_path
        )
    )

    data = pd.read_csv(args.input_csv)

    evidence = [eval(i) for i in data['evidence']]

    generated = []

    for reviews in tqdm(evidence):
        z_raw = vae.encode(reviews)
        zs = torch.stack([z_raw.mean(dim=0)]) 
        output = vae.generate(zs, bad_words=util.BAD_WORDS, num_beams=1)[0]  
        generated.append(output)

    data['vae_generated'] = generated

    data.to_csv(args.output_csv)


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
                    optimus_path='coop-finetune/log/optimus_1_epoch_0.5_beta/tripadvisor/ex1/model-step_32009.th',
                    output_csv='./tripadvisor_generated.csv'
                )
            )
        elif args.auto_arg_by_dataset == 'yelp':
            args = edict(
                dict(
                    input_csv='prag_generation/generated/yelp_generated.csv',
                    optimus_path='coop-finetune/log/optimus_1_epoch_0.5_beta/yelp/ex1/model-step_130000.th',
                    output_csv='./yelp_generated.csv'
                )
            )
        elif args.auto_arg_by_dataset == 'movies_and_tv':
            args = edict(
                dict(
                    input_csv='prag_generation/generated/movies_and_tv_generated.csv',
                    optimus_path='coop-finetune/log/optimus_1_epoch_0.5_beta/movies_and_tv/ex1/model-step_44190.th',
                    output_csv='./movies_and_tv_generated.csv'
                )
            )
    main(args)