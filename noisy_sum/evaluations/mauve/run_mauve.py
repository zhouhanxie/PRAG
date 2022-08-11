import mauve 
import sys
sys.path.insert(0, 'personalized_retriever')
from utils import DataReader
import pandas as pd

def clean(sent):
    """
    heuristic truncation that clips out
    things after the last period
    """
    try:
        cleaned_sent = sent.split('.')
        if len(cleaned_sent) > 1:
            cleaned_sent = cleaned_sent[:-1]
            
        out =  '.'.join(cleaned_sent)
        assert out != ''
    except AssertionError:
        print('error occured')
        return 'none'
    return out

def main(args):

    data = DataReader(
                data_path = args.data_path,
                index_dir = args.index_dir
            )

    generated = pd.read_csv(args.csv_dir)['summarizer_generated'].tolist()
    generated = [eval(i)['summary'] for i in generated]
    generated = [i if len(i)>0 else '<|endoftext|>' for i in generated]
    gold = [i['text'] for i in data.test[:len(generated)]]
    
    
    out = mauve.compute_mauve(
        p_text=[clean(i) for i in generated], 
        q_text=gold, 
        device_id=0, 
        max_text_length=60, 
        verbose=True
    )
    print(out.mauve) 

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
                    data_path='../../../nlg4rec_data/TripAdvisor/reviews.pickle', 
                    index_dir='../../../nlg4rec_data/TripAdvisor/1',
                    csv_dir='noisy_sum/generated/tripadvisor_generated.csv'
                )
            )
        elif args.auto_arg_by_dataset == 'yelp':
            args = edict(
                dict(
                    data_path='../../../nlg4rec_data/Yelp/reviews.pickle', 
                    index_dir='../../../nlg4rec_data/Yelp/1',
                    csv_dir='noisy_sum/generated/yelp_generated.csv'
                )
            )
        elif args.auto_arg_by_dataset == 'movies_and_tv':
            args = edict(
                dict(
                    data_path='../../../nlg4rec_data/Amazon/MoviesAndTV/reviews.pickle', 
                    index_dir='../../../nlg4rec_data/Amazon/MoviesAndTV/1',
                    csv_dir='noisy_sum/generated/movies_and_tv_generated.csv'
                )
            )
    main(args)