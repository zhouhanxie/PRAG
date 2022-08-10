import torch

class RetrievalPipeline(torch.nn.Module):
    
    def __init__(self, retriever, review_history):
        super().__init__()
        self.retriever =  retriever
        self.review_history = review_history
        assert torch.cuda.is_available()
        self.device = torch.device('cuda')
        self.to(self.device)
        
    def prepare_model_input(self, userid, itemid):
        """
        unbatched assemble of retriever input
        """
        device = self.device
        review_history = self.review_history
        
        user = torch.tensor([userid])
        item = torch.tensor([itemid])
        review_history_input_ids = []
        review_history_token_type_ids = []
        review_history_input_ids += review_history.get_user_history(
            user=userid, hide_item=itemid, return_embedding=False
        )
        while len(review_history_token_type_ids) < len(review_history_input_ids):
            review_history_token_type_ids.append(0)
        review_history_input_ids += review_history.get_item_history(
            item=itemid, hide_user=userid, return_embedding=False
        )

        while len(review_history_token_type_ids) < len(review_history_input_ids):
            review_history_token_type_ids.append(1)

        output = (
            torch.atleast_1d(user).to(device), 
            torch.atleast_1d(item).to(device),
            torch.atleast_2d(torch.tensor(review_history_input_ids)).to(device), 
            torch.atleast_2d(torch.tensor(review_history_token_type_ids)).to(device), 
            torch.atleast_2d(torch.ones(len(review_history_input_ids)).int()).to(device)
        )

        return output
    
    def retrieve_review(
        self, 
        user, 
        item, 
        N=10, 
        return_embedding_only = False, 
        marginalize=False, 
        return_adjustment=False, 
        filter_by_adjustment=True,
        return_document_embedding=False
        ):
        """
        unbatched review retrieval, returns list of evidence retrieved, and optionally the adjust term
        """
        device=self.device 
        retriever = self.retriever
        review_history = self.review_history

        with torch.no_grad():
            user, item, review_history_input_ids, review_history_token_type_ids, review_history_attention_mask = \
            self.prepare_model_input(user, item)

            # print(review_history_attention_mask)

            review_embedding = retriever.predict_review_embedding(
                user = user, 
                item = item,
                review_history_input_ids = review_history_input_ids,
                review_history_token_type_ids = review_history_token_type_ids,
                review_history_attention_mask = review_history_attention_mask
            )

            if return_embedding_only and marginalize == False:
                return review_embedding

            if self.retriever.config.use_user_topic:
                adjustment =(
                    retriever.rating_regression_head.rating_downproj(review_embedding) 
                    * 
                    retriever.rating_regression_head.item_factor[item]
                    ).sum(dim=-1).cpu().item()
            else:
                adjustment =(
                    retriever.rating_regression_head.rating_downproj(review_embedding) 
                    * 
                    retriever.rating_regression_head.user_factor[user]
                    ).sum(dim=-1).cpu().item()

            review_embedding = review_embedding.cpu()

            factoids = torch.from_numpy(review_history.text_table[review_history_input_ids.cpu().numpy()])

            # estimate the embedding that marginalizes the user factor
            if marginalize == 'item':
                sampled_embeddings = []
                for sampled_item in torch.randint(low=0, high=5000, size=[50]):
                    sampled_review_embedding = retriever.predict_review_embedding(
                        user = user, 
                        item = sampled_item,
                        review_history_input_ids = review_history_input_ids,
                        review_history_token_type_ids = review_history_token_type_ids,
                        review_history_attention_mask = review_history_attention_mask
                    ).cpu()
                    sampled_embeddings.append(sampled_review_embedding)
                mean_sampled_embeddings = torch.cat(sampled_embeddings, dim=0)
                mean_sampled_embeddings = mean_sampled_embeddings.mean(dim=0)
                review_embedding = review_embedding - mean_sampled_embeddings
                if return_embedding_only:
                    return review_embedding
                retrieval_loss = -torch.nn.functional.cosine_similarity(
                    review_embedding.unsqueeze(-2).repeat(1, factoids.shape[1],1), 
                    factoids,
                    dim=-1
                ).cpu()

            elif marginalize == 'user':
                sampled_embeddings = []
                for sampled_user in torch.randint(low=0, high=5000, size=[50]):
                    sampled_review_embedding = retriever.predict_review_embedding(
                        user = sampled_user, 
                        item = item,
                        review_history_input_ids = review_history_input_ids,
                        review_history_token_type_ids = review_history_token_type_ids,
                        review_history_attention_mask = review_history_attention_mask
                    ).cpu()
                    sampled_embeddings.append(sampled_review_embedding)
                mean_sampled_embeddings = torch.cat(sampled_embeddings, dim=0)
                mean_sampled_embeddings = mean_sampled_embeddings.mean(dim=0)
                review_embedding = review_embedding - mean_sampled_embeddings
                if return_embedding_only:
                    return review_embedding
                retrieval_loss = -torch.nn.functional.cosine_similarity(
                    review_embedding.unsqueeze(-2).repeat(1, factoids.shape[1],1), 
                    factoids,
                    dim=-1
                ).cpu()


            else:
                retrieval_loss = torch.nn.functional.mse_loss(
                        input = review_embedding.unsqueeze(-2).repeat(1, factoids.shape[1],1), 
                        target = factoids,
                        reduction='none'
                    ).sum(dim=-1).cpu()
            sorted_idx_by_loss = torch.argsort(retrieval_loss)

            sorted_facts_idx = review_history_input_ids.flatten()[sorted_idx_by_loss.flatten()].cpu().numpy()

            # filter by evidence's sentiment
            if filter_by_adjustment:
                # print('filtering...')
                rating_of_sorted_idx = self.review_history.rating_table[sorted_facts_idx]
                if adjustment > 0:
                    sorted_facts_idx = sorted_facts_idx[rating_of_sorted_idx >= int(self.review_history.mean_rating())]
                else:
                    sorted_facts_idx = sorted_facts_idx[rating_of_sorted_idx < int(self.review_history.mean_rating())]
            
            # could get text or embedding
            if not return_document_embedding:
                evidence = review_history.raw_text_table[sorted_facts_idx][:N].tolist()
            elif return_document_embedding == 'both_text_and_emb':
                evidence = (
                    review_history.raw_text_table[sorted_facts_idx][:N].tolist() 
                    , 
                    review_history.text_table[sorted_facts_idx][:N]
                    )
            else:
                evidence = review_history.text_table[sorted_facts_idx][:N]
            
            evidence_rating = review_history.rating_table[sorted_facts_idx][:N]
            
            if return_adjustment:
                return evidence, adjustment
            else:
                return evidence