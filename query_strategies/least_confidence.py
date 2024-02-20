import numpy as np
import torch
import random
from .strategy import Strategy
# class LeastConfidence(Strategy):
#     def __init__(self, dataset, net):
#         super(LeastConfidence, self).__init__(dataset, net)

#     def query(self, n, current_round, js_divergence):
#         unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
#         probs = self.predict_prob(unlabeled_data)

#         uncertainties = probs.sum((1)) # probs.sum((1,2,3)).shape ([7250])
#         return unlabeled_idxs[uncertainties.sort()[1][:n]]
    

from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
from config import parse_args
args = parse_args()

class LeastConfidence(Strategy):
    def __init__(self, dataset, net):
        super(LeastConfidence, self).__init__(dataset, net)
        self.total_rounds = args.n_round
        # self.js_divergence_threshold = 0.1  # 设置JS散度的阈值
        self.use_uncertainty = False  # 添加一个状态标记

    def query(self, n, current_round, js_divergence,param2,param3):
        if param3 != None:
            unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
            probs = self.predict_prob(unlabeled_data)

            uncertainties = probs.sum((1)) # probs.sum((1,2,3)).shape ([7250])
            sorted_indices = uncertainties.sort()[1]
            

            levels = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

            if js_divergence < float(param2) or self.use_uncertainty:
                # 如果JS散度低于阈值，专注于最难的样本
                self.use_uncertainty = True
                selected = []
                selected = sorted_indices[:n]
                selected_indices = unlabeled_idxs[selected]

                # # 从第一个难度范围选择90%的样本
                # level_start_index = int(len(sorted_indices) * levels[0][0])
                # level_end_index = int(len(sorted_indices) * levels[0][1])
                # level_indices = sorted_indices[level_start_index:level_end_index]
                # selected += list(np.random.choice(level_indices, int(n), replace=False))

                # selected += list(np.random.choice(level_indices, int(n * 0.9), replace=False))

                # 从第二个难度范围选择10%的样本
                # level_start_index = int(len(sorted_indices) * levels[1][0])
                # level_end_index = int(len(sorted_indices) * levels[1][1])
                # level_indices = sorted_indices[level_start_index:level_end_index]
                # selected += list(np.random.choice(level_indices, int(n * 0.1), replace=False))

                # selected_indices = unlabeled_idxs[selected]

            else:
                # Random
                selected = []
                # print("bbbbbbb")

                # k = len(unlabeled_idxs) // n
                
                # # Select indices using calculated 'k'
                # selected_indices = unlabeled_idxs[::k][:n]
                if param3 == "kmedoid+hard":
                # samples_per_level = [20, 20, 20, 20, 20]
                    scaler = StandardScaler()

                    embeddings = self.get_embeddings(unlabeled_data)
                    embeddings = scaler.fit_transform(embeddings.numpy())  # 标准化数据
                    # print("embeddings", embeddings.shape)

                #     # KMedoids聚类，尝试不同的距离度量和更多迭代
                    cluster_learner = KMedoids(n_clusters=n, metric='euclidean', max_iter=300, init='k-medoids++')
                    cluster_learner.fit(embeddings)

                    medoid_indices = cluster_learner.medoid_indices_

                    selected += list(unlabeled_idxs[medoid_indices])
                    selected_indices = selected

                elif param3 == "uncertainty_random+hard":
                    samples_per_level = [20, 20, 20, 20, 20]
                    selected = []
                    for idx, level in enumerate(levels):
                        start, end = int(level[0] * len(sorted_indices)), int(level[1] * len(sorted_indices))
                        selected += random.sample(list(sorted_indices[start:end]), samples_per_level[idx])
                    selected_indices = unlabeled_idxs[selected]

                # unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
                elif param3 == "random+hard" or param3 == "random":
                    selected_indices = np.random.choice(unlabeled_idxs, n, replace=False)

            return selected_indices
        
        else: 
            unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
            probs = self.predict_prob(unlabeled_data)

            uncertainties = probs.sum((1)) # probs.sum((1,2,3)).shape ([7250])
            return unlabeled_idxs[uncertainties.sort()[1][:n]]