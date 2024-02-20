import numpy as np
import torch
from .strategy import Strategy

# # Use the prediction entropy as uncertainty
# class EntropySampling(Strategy):
#     def __init__(self, dataset, net):
#         super(EntropySampling, self).__init__(dataset, net)

#     # def query(self, n, index, pred):
#         # log_probs = torch.log(pred)#([12384, 1, 128, 128])
#         # uncertainties = (pred*log_probs).sum((1,2,3))#([12384])
#         # return index[uncertainties.sort()[1][:n]]

#     def query(self, n, current_round, js_divergence):
#         unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
#         probs = self.predict_prob(unlabeled_data)
#         log_probs = torch.log(probs)
#         uncertainties = (probs*log_probs).sum((1))#([12384])
#         return unlabeled_idxs[uncertainties.sort()[1][:n]]
    
import torch
import random
from .strategy import Strategy
from config import parse_args
args = parse_args()

# v1-round 
# class EntropySampling(Strategy):
#     def __init__(self, dataset, net):
#         super(EntropySampling, self).__init__(dataset, net)
#         self.total_rounds = args.n_round

#     def query(self, n, current_round):
#         unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
#         probs = self.predict_prob(unlabeled_data)
#         log_probs = torch.log(probs)
#         uncertainties = (probs * log_probs).sum(1)  # Calculate the entropy

#         sorted_indices = uncertainties.sort()[1]

#         # Define percentage ranges for easy, medium, and hard samples
#         easy_range = (0.7, 0.95)   # Select from the easiest 30%
#         medium_range = (0.3, 0.7) # Select from the middle 40%
#         hard_range = (0.05, 0.3)   # Select from the hardest 30%

#         if current_round < self.total_rounds / 3:
#             # Early rounds: select from the easier part
#             start, end = int(easy_range[0] * len(sorted_indices)), int(easy_range[1] * len(sorted_indices))

#         elif current_round < 2 * self.total_rounds / 3:
#             # Middle rounds: select from the medium part
#             start, end = int(medium_range[0] * len(sorted_indices)), int(medium_range[1] * len(sorted_indices))

#         else:
#             # Later rounds: select from the harder part
#             start, end = int(hard_range[0] * len(sorted_indices)), int(hard_range[1] * len(sorted_indices))

#         # Randomly select samples from the specified range to avoid redundancy
#         selected = random.sample(list(sorted_indices[start:end]), n)

#         return unlabeled_idxs[selected]

# def update_uncertainty_threshold(self, current_performance, initial_threshold=0.5):
#     """
#     根据当前模型性能动态调整不确定性阈值
#     :param current_performance: 当前模型在验证集上的性能
#     :param initial_threshold: 初始阈值
#     :return: 调整后的阈值
#     """
#     if current_performance > self.previous_performance:
#         # 模型表现改善，增加阈值
#         new_threshold = min(initial_threshold + 0.1, 1.0)
#     else:
#         # 模型表现停滞或退步，降低阈值
#         new_threshold = max(initial_threshold - 0.1, 0.0)
#     self.previous_performance = current_performance
#     return new_threshold


# class EntropySampling(Strategy):
#     def __init__(self, dataset, net):
#         super(EntropySampling, self).__init__(dataset, net)
#         self.total_rounds = args.n_round

#     def query(self, n, current_round):
#         unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
#         labeled_idxs, labeled_data = self.dataset.get_train_data()

#         probs = self.predict_prob(unlabeled_data)
#         log_probs = torch.log(probs)
#         uncertainties = (probs * log_probs).sum(1) 

#         l_probs = self.predict_prob(labeled_data)
#         l_log_probs = torch.log(l_probs)
#         labeled_uncertainty = (l_probs * l_log_probs).sum(1) 

#         # 计算最大值和最小值
#         uncertainty_min = torch.min(labeled_uncertainty)
#         uncertainty_max = torch.max(labeled_uncertainty)

#         # 应用最小-最大归一化
#         labeled_uncertainty = (labeled_uncertainty - uncertainty_min) / (uncertainty_max - uncertainty_min)
#         labeled_uncertainty = torch.mean(labeled_uncertainty)

#         sorted_indices = uncertainties.sort()[1]

#         # Define ranges for easy, medium, and hard samples
#         ranges = [
#             (0.0, 0.2),  # Very easy
#             (0.2, 0.4),  # Easy
#             (0.4, 0.6),  # Medium
#             (0.6, 0.8),  # Hard
#             (0.8, 1.0)   # Very hard
#         ]

#         # Adjust range based on labeled_uncertainty
#         uncertainty_threshold = 0.5 # Define a threshold for uncertainty 这里可以是动态的
#         if labeled_uncertainty < uncertainty_threshold:
#             # If uncertainty is low, choose harder samples earlier
#             range_idx = min(current_round * len(ranges) // self.total_rounds, len(ranges) - 2)
#         else:
#             # If uncertainty is high, choose easier samples longer
#             range_idx = min(current_round * len(ranges) // self.total_rounds, len(ranges) - 3)

#         # Select samples from the appropriate range
#         start, end = ranges[range_idx]
#         start_idx, end_idx = int(start * len(sorted_indices)), int(end * len(sorted_indices))

#         # Randomly select samples from the specified range to avoid redundancy
#         selected = random.sample(list(sorted_indices[start_idx:end_idx]), n)

#         return unlabeled_idxs[selected]


# class EntropySampling(Strategy):
#     def __init__(self, dataset, net):
#         super(EntropySampling, self).__init__(dataset, net)
#         self.total_rounds = args.n_round
#         self.previous_val_loss = float('inf')

#     def query(self, n, current_round, current_val_loss):
#         unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
#         labeled_idxs, labeled_data = self.dataset.get_train_data()

#         # Calculate uncertainties
#         probs = self.predict_prob(unlabeled_data)
#         log_probs = torch.log(probs)
#         uncertainties = (probs * log_probs).sum(1) 

#         l_probs = self.predict_prob(labeled_data)
#         l_log_probs = torch.log(l_probs)
#         labeled_uncertainty = (l_probs * l_log_probs).sum(1) 
#         uncertainty_min = torch.min(labeled_uncertainty)
#         uncertainty_max = torch.max(labeled_uncertainty)
#         labeled_uncertainty = (labeled_uncertainty - uncertainty_min) / (uncertainty_max - uncertainty_min)
#         labeled_uncertainty = torch.mean(labeled_uncertainty)

#         sorted_indices = uncertainties.sort()[1]
#         ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

#         # Dynamic adjustment based on validation loss and labeled uncertainty
#         if current_val_loss < self.previous_val_loss and labeled_uncertainty < 0.5:
#             # Performance improved and low uncertainty: choose harder samples
#             range_idx = min(current_round * len(ranges) // self.total_rounds, len(ranges) - 1)
#         elif current_val_loss >= self.previous_val_loss and labeled_uncertainty >= 0.5:
#             # Performance decreased and high uncertainty: choose easier samples
#             range_idx = 0
#         else:
#             # Default or intermediate scenario
#             range_idx = min(current_round * len(ranges) // self.total_rounds, len(ranges) - 2)

#         start, end = ranges[range_idx]
#         start_idx, end_idx = int(start * len(sorted_indices)), int(end * len(sorted_indices))
#         selected = random.sample(list(sorted_indices[start_idx:end_idx]), n)

#         self.previous_val_loss = current_val_loss
#         return unlabeled_idxs[selected]


# 加入多样性
# distance_to_labeled = compute_distance(unlabeled_data, labeled_data)

# # 然后结合不确定性和距离信息选择样本
# selected_indices = []
# for start, end in [(start_easy, end_easy), (start_medium, end_medium), (start_hard, end_hard)]:
#     # 在每个难度范围内选择样本
#     range_indices = sorted_indices[start:end]
#     # 根据距离选择样本
#     range_indices = sorted(range_indices, key=lambda idx: distance_to_labeled[idx], reverse=True)
#     selected_indices.extend(range_indices[:n_per_range])  # 假设 n_per_range 是每个范围内要选择的样本数

# selected_samples = unlabeled_idxs[selected_indices]


# class EntropySampling(Strategy):
#     def __init__(self, dataset, net):
#         super(EntropySampling, self).__init__(dataset, net)
#         self.total_rounds = args.n_round

#     def query(self, n, current_round):
#         unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
#         probs = self.predict_prob(unlabeled_data)
#         log_probs = torch.log(probs)
#         uncertainties = (probs * log_probs).sum(1)  # Calculate the entropy

#         sorted_indices = uncertainties.sort()[1]

#         # Define percentage ranges for easy, medium, and hard samples
#         easy_range = (0.8, 0.98)   # Select from the easiest 30%
#         # medium_range = (0.3, 0.7) # Select from the middle 40%
#         # hard_range = (0.05, 0.3)   # Select from the hardest 30%

#         if current_round < self.total_rounds / 5:
#             # Early rounds: select from the easier part
#             start, end = int(easy_range[0] * len(sorted_indices)), int(easy_range[1] * len(sorted_indices))
#             selected = random.sample(list(sorted_indices[start:end]), n)
#             return unlabeled_idxs[selected]

#         else:
#             # Later rounds: select from the harder part
#             return unlabeled_idxs[uncertainties.sort()[1][:n]]

# import torch

# class EntropySampling(Strategy):
#     def __init__(self, dataset, net):
#         super(EntropySampling, self).__init__(dataset, net)
#         self.total_rounds = args.n_round

#     def query(self, n, current_round):
#         unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
#         probs = self.predict_prob(unlabeled_data)
#         log_probs = torch.log(probs)
#         uncertainties = (probs * log_probs).sum(1)  # Calculate the entropy

#         sorted_indices = uncertainties.sort()[1]

#         # 使用非线性函数调整难学样本的比例
#         # 比如这里使用了一个简单的二次函数
#         hard_sample_percentage = (current_round / self.total_rounds) ** 2
#         easy_sample_percentage = 1 - hard_sample_percentage

#         # Calculate the number of easy and hard samples to be selected
#         n_easy = int(n * easy_sample_percentage)
#         n_hard = n - n_easy

#         # Select easy samples (lowest uncertainty)
#         easy_samples = sorted_indices[-n_easy:]

#         # Select hard samples (highest uncertainty)
#         hard_samples = sorted_indices[:n_hard]

#         # Combine selected samples
#         selected = torch.cat((easy_samples, hard_samples)).tolist()

#         return unlabeled_idxs[selected]

# import torch
# import random

# import torch
# import random

# class EntropySampling(Strategy):
#     def __init__(self, dataset, net):
#         super(EntropySampling, self).__init__(dataset, net)
#         self.total_rounds = args.n_round

#     def query(self, n, current_round):
#         unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
#         probs = self.predict_prob(unlabeled_data)
#         log_probs = torch.log(probs)
#         uncertainties = (probs * log_probs).sum(1)  # Calculate the entropy

#         sorted_indices = uncertainties.sort()[1]

#         # Define percentage ranges for different difficulty levels
#         levels = [
#             (0.0, 0.2),    # Very hard
#             (0.2, 0.4),    # Hard
#             (0.4, 0.6),    # Medium
#             (0.6, 0.8),    # Easy
#             (0.8, 1.0)     # Very easy
#         ]

#         # Adjust the number of samples to select from each level based on the current round
#         progress = current_round / self.total_rounds  # Progress ratio
#         # More weight to harder levels as rounds progress
#         level_weights = [1 + (len(levels) - 1 - i) * progress for i in range(len(levels))]
#         total_weight = sum(level_weights)
#         samples_per_level = [int(n * (weight / total_weight)) for weight in level_weights]
#         print(samples_per_level)

#         # Adjust for rounding errors
#         while sum(samples_per_level) != n:
#             samples_per_level[-1] += n - sum(samples_per_level)

#         # samples_per_level = [20,20,20,20,20]
#         selected = []
#         for idx, level in enumerate(levels):
#             start, end = int(level[0] * len(sorted_indices)), int(level[1] * len(sorted_indices))
#             selected += random.sample(list(sorted_indices[start:end]), samples_per_level[idx])

#         return unlabeled_idxs[selected]

# class EntropySampling(Strategy):
#     def __init__(self, dataset, net):
#         super(EntropySampling, self).__init__(dataset, net)
#         self.total_rounds = args.n_round

#     def query(self, n, current_round):
#         unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
#         probs = self.predict_prob(unlabeled_data)
#         log_probs = torch.log(probs)
#         uncertainties = (probs * log_probs).sum(1)  # Calculate the entropy

#         sorted_indices = uncertainties.sort()[1]

#         # Define percentage ranges for easy, medium, and hard samples
#         easy_range = (0.8, 0.98)   # Select from the easiest 30%
#         # medium_range = (0.3, 0.7) # Select from the middle 40%
#         # hard_range = (0.05, 0.3)   # Select from the hardest 30%

#         if current_round < self.total_rounds / 5:
#             # Early rounds: select from the easier part
#             start, end = int(easy_range[0] * len(sorted_indices)), int(easy_range[1] * len(sorted_indices))
#             selected = random.sample(list(sorted_indices[start:end]), n)
#             return unlabeled_idxs[selected]

#         else:
#             # Later rounds: select from the harder part
#             return np.random.choice(unlabeled_idxs, n, replace=False)
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class EntropySampling(Strategy):
    def __init__(self, dataset, net):
        super(EntropySampling, self).__init__(dataset, net)
        self.total_rounds = args.n_round
        # self.js_divergence_threshold = 0.1  # 设置JS散度的阈值
        self.use_uncertainty = False  # 添加一个状态标记

    def query(self, n, current_round, js_divergence,param2,param3):
        # if param3 == "random+hard":
        if param3 != None:

            unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
            probs = self.predict_prob(unlabeled_data)
            log_probs = torch.log(probs)
            uncertainties = (probs * log_probs).sum(1)
            sorted_indices = uncertainties.sort()[1]
            
            levels = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

            if js_divergence < float(param2) or self.use_uncertainty:
                # print("aaaaaaa")
                # 如果JS散度低于阈值，专注于最难的样本
                self.use_uncertainty = True
                selected = []
                selected = sorted_indices[:n]
                selected_indices = unlabeled_idxs[selected]

                # 从第一个难度范围选择90%的样本
                # level_start_index = int(len(sorted_indices) * levels[0][0])
                # level_end_index = int(len(sorted_indices) * levels[0][1])
                # level_indices = sorted_indices[level_start_index:level_end_index]
                # selected += list(np.random.choice(level_indices, int(n * 0.9), replace=False))

                # # 从第二个难度范围选择10%的样本
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

                elif param3 == "coreset+hard":
                    labeled_idxs, train_data = self.dataset.get_train_data()
                    embeddings = self.get_embeddings(train_data)
                    embeddings = embeddings.numpy()

                    dist_mat = np.matmul(embeddings, embeddings.transpose())
                    sq = np.array(dist_mat.diagonal()).reshape(len(labeled_idxs), 1)
                    dist_mat *= -2
                    dist_mat += sq
                    dist_mat += sq.transpose()
                    dist_mat = np.sqrt(dist_mat)

                    mat = dist_mat[~labeled_idxs, :][:, labeled_idxs]

                    for i in tqdm(range(n), ncols=100):
                        mat_min = mat.min(axis=1)
                        q_idx_ = mat_min.argmax()
                        q_idx = np.arange(self.dataset.n_pool)[~labeled_idxs][q_idx_]
                        labeled_idxs[q_idx] = True
                        mat = np.delete(mat, q_idx_, 0)
                        mat = np.append(mat, dist_mat[~labeled_idxs, q_idx][:, None], axis=1)
                        
                    selected_indices =  np.arange(self.dataset.n_pool)[(self.dataset.labeled_idxs ^ labeled_idxs)]

            return selected_indices
        
        else: 
            unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
            probs = self.predict_prob(unlabeled_data)
            log_probs = torch.log(probs)
            uncertainties = (probs*log_probs).sum((1))#([12384])
            return unlabeled_idxs[uncertainties.sort()[1][:n]]