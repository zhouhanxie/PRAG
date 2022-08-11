import sys
sys.path.insert(0, '../embedding_interpreter')
sys.path.insert(0, '../personalized_retriever')

import torch
import pandas as pd

from transformers import AutoTokenizer
from modeling_prag_decoder import CausalPragDecoderModel
from tqdm import tqdm
from retrieval_pipeline import RetrievalPipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
from huggingface_utils import move_to_device
from modeling_personalized_retriever import PersonalizedRetrieverModel
from utils import ReviewHistory, DataReader
from transformers import DisjunctiveConstraint

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

    # topic embedding decoder related stuff
    decoder = CausalPragDecoderModel.from_pretrained(args.embedding_interpreter_dir)
    decoder = decoder.cuda()
    decoder_tokenizer = AutoTokenizer.from_pretrained('gpt2') # trained with gpt2 vocab

    # prompt (NL)
    pos_prompt = "what was great ?"
    neg_prompt = "what was not good ?"
    general_prompt = "what was great and what was not good ?\\n"

    # generation loop
    all_generated = []
    c = 0
    for sample in tqdm(data.test[:args.max_sample]):
        c += 1
        if c%args.save_every == 0 or c==100:
            print('saving partial...')
            pd.DataFrame(all_generated).to_csv(args.output_dir)
        
        with torch.no_grad():
            query = retrieval_pipeline.retrieve_review(
                sample['user'], 
                sample['item'], 
                return_embedding_only=True
            ).cuda().squeeze()
            if retriever.config.use_user_topic:
                retrieved_text_and_emb, adj_term = retrieval_pipeline.retrieve_review(
                    sample['user'], 
                    sample['item'], 
                    marginalize='user', 
                    N=10,
                    return_document_embedding='both_text_and_emb',
                    return_adjustment=True
                )
            else:
                retrieved_text_and_emb, adj_term = retrieval_pipeline.retrieve_review(
                    sample['user'], 
                    sample['item'], 
                    marginalize='item', 
                    N=10,
                    return_document_embedding='both_text_and_emb',
                    return_adjustment=True
                )
            retrieved_text, retrieved_emb = retrieved_text_and_emb
                
            z = query+torch.tensor(retrieved_emb).mean(dim=0).cuda()
            tot_topic = decoder.inference(torch.atleast_2d( z ), tokenizer=decoder_tokenizer)
            tot_topic = tot_topic.replace('generated#', '').replace('<|endoftext|>', '').split(',')
            evidence = ' . '.join(retrieved_text)
            evidence_terms = set(evidence.split())
            tot_topic = [w.strip() for w in tot_topic]
            tot_topic = [w for w in tot_topic if w in evidence_terms]
            
            if adj_term >= 0:
                prompt = pos_prompt+'\\n'
            else:
                prompt = neg_prompt+'\\n'

            model_input_text = assemble_model_input(prompt, tot_topic, evidence)
            # print(model_input_text)
            model_input_text = model_input_text.lower()
            generated = run_model(
                model_input_text,
                num_beams=1, 
                min_length=5, 
                max_length=25, 
                no_repeat_ngram_size=0
            )[0]
            generated = generated.replace('.', '. ')
        
        data_structure = {
            'adj_term': adj_term, 
            'generated':generated, 
            'prompt':prompt, 
            'topic':tot_topic, 
            'evidence':retrieved_text
        }
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
                        help='petrained retriever')
    parser.add_argument('--embedding_interpreter_dir', type=str, default=None,
                        help='retrained embedding_interpreter')
    parser.add_argument('--cached_review_embedding', type=str, default=None,
                        help='embedding table')
    parser.add_argument('--output_dir', type=str, default='tmp.csv',
                        help='output')
    parser.add_argument('--save_every', type=int, default=5000,
                        help='report interval')
    parser.add_argument('--auto_arg_by_dataset', type=str, default=None,
                        help='auto argument')
    parser.add_argument('--max_sample', type=int, default=10000,
                        help='clip to maximum dp to generate')
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
                    embedding_interpreter_dir='/embedding_interpreter/TripAdvisor/1/checkpoint-68034',
                    output_dir='generated/tripadvisor_generated.csv',
                    save_every=5000,
                    max_sample=10000
                )
            )
        elif args.auto_arg_by_dataset == 'yelp':
            args = edict(
                dict(
                    data_path='../nlg4rec_data/Yelp/reviews.pickle', 
                    index_dir='../nlg4rec_data/Yelp/1',
                    cached_review_embedding='transformer_retriever/yelp_table.npy',
                    retriever_dir='personalized_retriever/mpnet_space/item_topic/Yelp/1/checkpoint-129328',
                    embedding_interpreter_dir='embedding_interpreter/Yelp/1/checkpoint-323320',
                    output_dir='generated/yelp_generated.csv',
                    save_every=5000,
                    max_sample=10000
                )
            )
        elif args.auto_arg_by_dataset == 'movies_and_tv':
            args = edict(
                dict(
                    data_path='../nlg4rec_data/Amazon/MoviesAndTV/reviews.pickle', 
                    index_dir='../nlg4rec_data/Amazon/MoviesAndTV/1',
                    cached_review_embedding='archive/transformer_retriever/amazon_moviesandtv_table.npy',
                    retriever_dir='personalized_retriever/mpnet_space/item_topic/Amazon/MoviesAndTV/1/checkpoint-77315',
                    embedding_interpreter_dir='embedding_interpreter/Amazon/MoviesAndTV/1/checkpoint-107718',
                    output_dir='generated/movies_and_tv_generated.csv',
                    save_every=5000,
                    max_sample=10000
                )
            )
    main(args)