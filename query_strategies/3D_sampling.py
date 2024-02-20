import numpy as np
import torch
from .strategy import Strategy

class EntropySampling(Strategy):
    def __init__(self, dataset, net):
        super(EntropySampling, self).__init__(dataset, net)

    # def query(self, n, blank_index, index, pred, handler):
    #     unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data(index = blank_index)
    #     probs = self.predict_prob(unlabeled_data) #([12384, 1, 128, 128])
    #     log_probs = torch.log(probs)#([12384, 1, 128, 128])
    #     uncertainties = (probs*log_probs).sum((1,2,3))#([12384])
    #     return unlabeled_idxs[uncertainties.sort()[1][:n]]

    def query(self, n, index, pred):
        log_probs = torch.log(pred)#([12384, 1, 128, 128])
        uncertainties = (pred*log_probs).sum((1,2,3))#([12384])
        return index[uncertainties.sort()[1][:n]]

def select_uncertain_and_low_similarity(dataset, labeled_data, model, k):
    # Calculate uncertainty scores for all unlabeled samples
    uncertainty_scores = calculate_uncertainty_scores(dataset, model)
    
    # Calculate similarity scores between unlabeled samples and labeled samples
    similarity_scores = calculate_similarity_scores(dataset, labeled_data, model)
    
    # Filter samples with high uncertainty and low similarity
    combined_scores = uncertainty_scores * (1 - similarity_scores) # Combine the two scores
    top_k_idxs = np.argsort(combined_scores)[-k:] # Get the top k indices with the highest combined scores
    
    return top_k_idxs
