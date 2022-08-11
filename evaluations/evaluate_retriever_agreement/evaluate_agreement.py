import sys
sys.path.insert(0, '../../personalized_retriever')

import torch
import numpy as np
np.random.seed(0)

from tqdm import tqdm
from easydict import EasyDict as edict
from retrieval_pipeline import RetrievalPipeline
from utils import ReviewHistory, DataReader
from modeling_personalized_retriever import PersonalizedRetrieverModel


def get_retrieval_pipeline(retriever_dir, review_history):
    """
    input:
        retriever_dir:
            dir to a HF model_name_or_path
            style dir of a PersonalizedRetrieverModel
        review_history:
            a ReviewHistory object
            whose pointer will be stored
    return:
        a RetrievalPipeline object
    """
    retriever = PersonalizedRetrieverModel.from_pretrained(retriever_dir)
    retrieval_pipeline = RetrievalPipeline(retriever, review_history)
    device = torch.device('cuda')
    retrieval_pipeline = retrieval_pipeline.to(device)
    return retrieval_pipeline


def random_item_reviews(review_history, sample, N):
    """
    review_history:
        a ReviewHistory object
    sample:
        a dictionary with
            'user' key -> user id
            'item' key -> item id
    returns:
        List[str]: sampled reviews
    """
    all_item_reviews_id = review_history.get_item_history(
        item=sample['item'], 
        hide_user=sample['user'], 
        return_embedding=False
    )
    selected_item_reviews_id = np.random.choice(
        all_item_reviews_id, 
        size=N, 
        replace=False
    )
    out = review_history.raw_text_table[selected_item_reviews_id]
    return out

def main(args):
    # load data
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

    # load models
    print('loading retrievers...')
    u_retriever = get_retrieval_pipeline(
        args.user_topic_retriever_dir,
        review_history
    )
    i_retriever = get_retrieval_pipeline(
        args.item_topic_retriever_dir,
        review_history
    )
    
    # evaluate agreement
    agreement_counts = []
    random_agreement_counts = []
    for sample in tqdm(data.test[:args.max_sample]):
        with torch.no_grad():

            u_retrieved_text = u_retriever.retrieve_review(
                sample['user'], 
                sample['item'], 
                marginalize=False, 
                N=args.N,
                return_document_embedding=False,
                return_adjustment=False
            )

            i_retrieved_text = i_retriever.retrieve_review(
                sample['user'], 
                sample['item'], 
                marginalize=False, 
                N=args.N,
                return_document_embedding=False,
                return_adjustment=False
            )

            agreement = set(u_retrieved_text).intersection(set(i_retrieved_text))
            agreement_count = len(agreement)
            agreement_counts.append(agreement_count)

            random_agreement = set(random_item_reviews(review_history, sample, args.N)).intersection(
                set(random_item_reviews(review_history, sample, args.N))
            )
            random_agreement_count = len(random_agreement)
            random_agreement_counts.append(random_agreement_count)


    # print results   
    agreemen_at_n = sum(agreement_counts)/len(agreement_counts)
    random_agreemen_at_n = sum(random_agreement_counts)/len(random_agreement_counts)
    print('agreement at ', args.N, ':', agreemen_at_n)
    print('random agreement at ', args.N, ':', random_agreemen_at_n)

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
                    data_path='../../nlg4rec_data/TripAdvisor/reviews.pickle', 
                    index_dir='../../nlg4rec_data/TripAdvisor/1',
                    cached_review_embedding='archive/transformer_retriever/tripadvisor_table.npy',
                    user_topic_retriever_dir='personalized_retriever/mpnet_space/user_topic/TripAdvisor/1/checkpoint-32004',
                    item_topic_retriever_dir='personalized_retriever/mpnet_space/item_topic/TripAdvisor/1/checkpoint-32004',
                    N=5,
                    max_sample=10000
                )
            )
        elif args.auto_arg_by_dataset == 'yelp':
            args = edict(
                dict(
                    data_path='../../nlg4rec_data/Yelp/reviews.pickle', 
                    index_dir='../../nlg4rec_data/Yelp/1',
                    cached_review_embedding='transformer_retriever/yelp_table.npy',
                    user_topic_retriever_dir='personalized_retriever/mpnet_space/user_topic/Yelp/1/checkpoint-129328',
                    item_topic_retriever_dir='personalized_retriever/mpnet_space/item_topic/Yelp/1/checkpoint-129328',
                    N=5,
                    max_sample=10000
                )
            )
        elif args.auto_arg_by_dataset == 'movies_and_tv':
            args = edict(
                dict(
                    data_path='../../nlg4rec_data/Amazon/MoviesAndTV/reviews.pickle', 
                    index_dir='../../nlg4rec_data/Amazon/MoviesAndTV/1',
                    cached_review_embedding='transformer_retriever/amazon_moviesandtv_table.npy',
                    user_topic_retriever_dir='personalized_retriever/mpnet_space/user_topic/Amazon/MoviesAndTV/1/checkpoint-99405',
                    item_topic_retriever_dir='personalized_retriever/mpnet_space/item_topic/Amazon/MoviesAndTV/1/checkpoint-77315',
                    N=5,
                    max_sample=10000
                )
            )
    main(args)
