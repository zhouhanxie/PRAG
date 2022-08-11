import torch
torch.__version__
from typing import List
import torch
import sys
sys.path.insert(0,'coop')
from coop import VAE, util
import nltk
from nltk.corpus import stopwords
stop_words_set = stopwords.words('english')

import rouge
# post hoc resolve extrinsic hallucination
# useful when doing convex aggregation
#     default behaviour is mean sum, do not use this heuristic
def input_output_overlap(inputs, output):
    r1 = rouge.Rouge(metrics=["rouge-n"], max_n=1, limit_length=False,)
    return r1.get_scores(output, inputs)["rouge-1"]["p"]


class TopicExtractionPipeline:
    
    def __init__(self, vae_name_or_path="megagonlabs/optimus-yelp"):
        # coop's vae has forced device loading (always cuda if available)
        self.vae = VAE("megagonlabs/optimus-yelp") # -> just a place holder model
        state_dict = torch.load(vae_name_or_path)
        self.vae.model.load_state_dict(state_dict)
        
        
    def generate_reviews(self, source_reviews, mean_sum=True, *generation_args, **generation_kwargs):
        """
        given a list of review, generate a target review.
        could be used as output as well, but bear in mind
        VAE usually has poor generation quality.
        """
        vae = self.vae
        z_raw: torch.Tensor = vae.encode(source_reviews)
        if mean_sum:
            zs = z_raw.mean(dim=0)
            outputs: List[str] = vae.generate(zs, num_beams=5, max_tokens=20)
            best = outputs[0]
        else:
            # convex aggregation
            # All combinations of input reviews
            idxes: List[List[int]] = util.powerset(len(source_reviews))
            # Taking averages for all combinations of latent vectors
            zs: torch.Tensor = torch.stack([z_raw[idx].mean(dim=0) for idx in idxes]) # [2^num_reviews - 1 * latent_size]
            outputs = []
            for cur_zs in chunkify(zs, 16):
                outputs += vae.generate(cur_zs, num_beams=10, max_tokens=20)
            # outputs: List[str] = vae.generate(zs, num_beams=5)
            best: str = max(outputs, key=lambda x: input_output_overlap(inputs=source_reviews, output=x))
        return best
    
    def generate_topic(self, source_reviews):
        """
        extract salient topic by interpolation source reviews.
        returns set of topic word.
        do not consider stop words.
        """
        generated = self.generate_reviews(
            source_reviews, 
            mean_sum=True
        )
        generated_set = set(generated.lower().split())
        retrieved_set = set(' '.join(source_reviews).lower().split())
        topic_words = retrieved_set.intersection(generated_set).difference(stop_words_set)
        return topic_words