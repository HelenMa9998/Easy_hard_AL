import numpy as np
from .strategy import Strategy
import torch.nn.functional as F
import torch

class CDALSampling(Strategy):
    def __init__(self, dataset, net):
        super(CDALSampling, self).__init__(dataset, net)
    
    def pairwise_distances(self, a, b):
        a = torch.from_numpy(a)
        b = torch.from_numpy(b)

        dist = np.zeros((a.size(0), b.size(0)), dtype=np.float)
        for i in range(b.size(0)):
            b_i = b[i]
            kl1 = a * torch.log(a / b_i)
            kl2 = b_i * torch.log(b_i / a)
            dist[:, i] = 0.5 * (torch.sum(kl1, dim=1)) + 0.5 * (torch.sum(kl2, dim=1))
        return dist

    def select_coreset(self, X, X_set, n):
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = self.pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        print('selecting coreset...')
        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = self.pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])
        return idxs
    
    def query(self, n, current_round, js_divergence,param2,param3):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs, embeddings = self.predict_prob_embed(unlabeled_data)
        labeled_idxs, labeled_data = self.dataset.get_cdal_labeled_data()
        probs_l, _ = self.predict_prob_embed(labeled_data)
        # print(n)
        # CDAL_CS coreset selection logic, needs to be defined
        # Assuming model has a method `select_coreset` which selects the core set based on the softmax probabilities
        chosen = self.select_coreset(probs.cpu().numpy(), probs_l.cpu().numpy(), n)
        return unlabeled_idxs[chosen]

    
	