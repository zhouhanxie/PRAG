import pandas as pd
from easydict import EasyDict as edict
import sys
sys.path.insert(0, '../embedding_interpreter')
sys.path.insert(0, '../personalized_retriever')
from utils import DataReader, ReviewHistory
import torch
from tqdm import tqdm
import numpy as np
import os
np.random.seed(0)
from pathlib import Path


def build_summarization_data(datalist, review_history, k=5, cos_sim_thresh=0.65):
    """
    build noisy summarization data
        by using similar reviews and randomly
        sampled ones as context/noise
    input:
        datalist: list of dict in the format like
                DataReader.train/valid/test
        review_history:
                ReviewHistory object
    return:
        list of dict with relevant fields
        see below for impl
    """

    summarization_data = []
    cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    for sample in tqdm(datalist):
        user, item = sample['user'], sample['item']
        target_review_id = review_history.get_ui_review(
            user = user,
            item = item,
            return_embedding=False
        )
        other_review_ids = torch.tensor(review_history.get_item_history(item=item, hide_user=user, return_embedding=False))
        cosine_sim = cosine_similarity(
                torch.from_numpy(review_history.text_table[target_review_id]).cuda(), 
                torch.from_numpy(review_history.text_table[other_review_ids]).cuda()
            ).cpu()
        top_k_cos_sim, top_k_indices = torch.topk(cosine_sim, k)
        least_match = top_k_cos_sim[-1]
        if least_match >= cos_sim_thresh:
            retrieved_context = review_history.raw_text_table[other_review_ids[top_k_indices]].tolist()
            target_review = sample['text']
            noise = np.random.choice(review_history.raw_text_table, size=5, replace=False).tolist()
            datum = {
                'retrieved_context':retrieved_context, 
                'target_review':target_review,
                'noise':noise
            } 
            datum['source'] = '<sep>'.join(
                np.random.permutation(datum['noise']+datum['retrieved_context']))
            summarization_data.append(datum)
    
    out = pd.DataFrame(summarization_data)
    print('num instance collected: ', len(out))
    return out

def main(args):
    Path(args.output_dir).mkdir( parents=True, exist_ok=True)
    data = DataReader(
        data_path = args.data_path,
        index_dir = args.index_dir
    )
    review_history = ReviewHistory(
        data.train, 
        valid_data=data.valid, 
        test_data=data.test
    ).build_embedded_text_table(
        None,
        None,
        maybe_load_from = args.cached_review_embedding
    )

    build_summarization_data(
        data.train, 
        review_history,
        k=5, 
        cos_sim_thresh=0.55
    ).to_csv(os.path.join(args.output_dir, 'train.csv'))

    build_summarization_data(
        data.valid, 
        review_history,
        k=5, 
        cos_sim_thresh=0.55
    ).to_csv(os.path.join(args.output_dir, 'valid.csv'))

    summarization_data_test = build_summarization_data(
        data.test, 
        review_history,
        k=5, 
        cos_sim_thresh=0.55
    ).to_csv(os.path.join(args.output_dir, 'test.csv'))


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
                    data_path='../nlg4rec_data/TripAdvisor/reviews.pickle', 
                    index_dir='../nlg4rec_data/TripAdvisor/1',
                    cached_review_embedding='archive/transformer_retriever/tripadvisor_table.npy',
                    retriever_dir='personalized_retriever/mpnet_space/item_topic/TripAdvisor/1/checkpoint-32004',
                    output_dir='noisy_summarization_data/tripadvisor/'
                )
            )
        elif args.auto_arg_by_dataset == 'yelp':
            args = edict(
                dict(
                    data_path='../nlg4rec_data/Yelp/reviews.pickle', 
                    index_dir='../nlg4rec_data/Yelp/1',
                    cached_review_embedding='archive/transformer_retriever/yelp_table.npy',
                    retriever_dir='personalized_retriever/mpnet_space/item_topic/Yelp/1/checkpoint-129328',
                    output_dir='noisy_summarization_data/yelp/'
                )
            )
        elif args.auto_arg_by_dataset == 'movies_and_tv':
            args = edict(
                dict(
                    data_path='../nlg4rec_data/Amazon/MoviesAndTV/reviews.pickle', 
                    index_dir='../nlg4rec_data/Amazon/MoviesAndTV/1',
                    cached_review_embedding='archive/transformer_retriever/amazon_moviesandtv_table.npy',
                    retriever_dir='personalized_retriever/mpnet_space/item_topic/Amazon/MoviesAndTV/1/checkpoint-77315',
                    output_dir='noisy_summarization_data/movies_and_tv/'
                )
            )
    main(args)


