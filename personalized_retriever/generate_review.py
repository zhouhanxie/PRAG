import torch
import pandas as pd

from tqdm import tqdm
from retrieval_pipeline import RetrievalPipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
from huggingface_utils import move_to_device
from modeling_personalized_retriever import PersonalizedRetrieverModel
from utils import ReviewHistory, DataReader
from transformers import DisjunctiveConstraint


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

    # retriever related stuff
    retriever = PersonalizedRetrieverModel.from_pretrained(args.retriever_dir)
    retrieval_pipeline = RetrievalPipeline(retriever, review_history)
    device = torch.device('cuda')
    retrieval_pipeline = retrieval_pipeline.to(device)

    # generator related stuff
    model_name = "allenai/unifiedqa-t5-3b" 
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    def run_model(input_string, **generator_args):
        model_input = tokenizer(input_string, return_tensors="pt", max_length=512, truncation=True)
        model_input = move_to_device(dict(model_input), device)
        res = model.generate(**model_input, **generator_args)
        return tokenizer.batch_decode(res, skip_special_tokens=True)

    # set up generation constraints
    force_words_ids = list(
        tokenizer(['great','good','wonderful','amazing','bad','horrible','the','was','were','is','are'],  
                add_special_tokens=False).input_ids
    )
    disjunctive_constraint = DisjunctiveConstraint(force_words_ids)
    pos_prompt = "what was great ?\\n "
    neg_prompt = "what was not good ?\\n "

    # generation loop
    all_generated = []
    c = 0
    for sample in tqdm(data.test):
        c += 1
        if c%args.save_every == 0 or c==100:
            print('saving partial...')
            pd.DataFrame(all_generated).to_csv(args.output_dir)
        with torch.no_grad():
            if retriever.config.use_user_topic:
                retrieved = retrieval_pipeline.retrieve_review(sample['user'], sample['item'], marginalize='user', N=8)
            else:
                retrieved = retrieval_pipeline.retrieve_review(sample['user'], sample['item'], marginalize='item', N=8)

            retrieved_marg , adj_term = retrieval_pipeline.retrieve_review(sample['user'], sample['item'], N=8, return_adjustment=True)

            if adj_term >= 0:
                prompt = pos_prompt
            else:
                prompt = neg_prompt
            generated = run_model(prompt+'. '.join(retrieved+retrieved_marg).lower(),num_beams=5, min_length=5, max_length=20, constraints=[disjunctive_constraint])[0]
            generated = generated.replace('.', '. ')
        data_structure = {'adj_term': adj_term, 'generated':generated, 'evidence':retrieved+retrieved_marg}
        all_generated.append(data_structure)
        
    # save
    print('saving all...')
    pd.DataFrame(all_generated).to_csv(args.output_dir)

if __name__ == '__main__':
    # from easydict import EasyDict as edict
    import argparse
    parser = argparse.ArgumentParser(description='Personalized Retriever model inference')
    parser.add_argument('--data_path', type=str, default=None,
                        help='path for loading the pickle data')
    parser.add_argument('--index_dir', type=str, default=None,
                        help='load indexes')
    parser.add_argument('--retriever_dir', type=str, default=None,
                        help='retrained retriever')
    parser.add_argument('--cached_review_embedding', type=str, default=None,
                        help='embedding table')
    parser.add_argument('--output_dir', type=str, default='tmp.csv',
                        help='output')
    parser.add_argument('--save_every', type=int, default=5000,
                        help='report interval')
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
                    cached_review_embedding='transformer_retriever/tripadvisor_table.npy',
                    retriever_dir='personalized_retriever/item_topic/TripAdvisor/1/checkpoint-32004',
                    output_dir='generated/tripadvisor_generated.csv',
                    save_every=5000
                )
            )
        elif args.auto_arg_by_dataset == 'yelp':
            args = edict(
                dict(
                    data_path='../nlg4rec_data/Yelp/reviews.pickle', 
                    index_dir='../nlg4rec_data/Yelp/1',
                    cached_review_embedding='transformer_retriever/yelp_table.npy',
                    retriever_dir='personalized_retriever/item_topic/Yelp/1/checkpoint-129328',
                    output_dir='generated/yelp_generated.csv',
                    save_every=5000
                )
            )
        elif args.auto_arg_by_dataset == 'movies_and_tv':
            args = edict(
                dict(
                    data_path='../nlg4rec_data/Amazon/MoviesAndTV/reviews.pickle', 
                    index_dir='../nlg4rec_data/Amazon/MoviesAndTV/1',
                    cached_review_embedding='transformer_retriever/amazon_moviesandtv_table.npy',
                    retriever_dir='personalized_retriever/item_topic/Amazon/MoviesAndTV/1/checkpoint-77315',
                    output_dir='generated/movies_and_tv_generated.csv',
                    save_every=5000
                )
            )
    main(args)