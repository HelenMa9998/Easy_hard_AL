import numpy as np
from .strategy import Strategy
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
from sklearn.cluster import AgglomerativeClustering
distance_threshold = 8

class ClusterMarginSampling(Strategy):
    def __init__(self, dataset, net):
        super(ClusterMarginSampling, self).__init__(dataset, net)
        # 初始化必要的参数或模型组件
        self.one_sample_step = True
    
    def margin_data(self, n):
        # unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        # probs = self.predict_prob(unlabeled_data).sum((1))
        # probs_sorted, _ = probs.sort(descending=True)
        # uncertainties = probs_sorted - (1-probs_sorted)#([7250])
        # return unlabeled_idxs[uncertainties.sort()[1][:n]]
        
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data).sum((1)) # 形状假设为 [N, C, H, W]
        probs_sorted, _ = probs.sort(descending=True) # 按类别概率排序，形状变为 [N, C, H, W]
        uncertainties = probs_sorted - (1-probs_sorted)#([7250])
        return unlabeled_idxs[uncertainties.sort(descending=True)[1][:n]]

    def round_robin(self, unlabeled_index, hac_list, k):
        cluster_list = []
        # print("Round Robin")
        for i in range(len(unlabeled_index)):
            cluster = []
            cluster_list.append(cluster)
        for real_idx in unlabeled_index:

            i = hac_list[real_idx]
            cluster_list[i].append(real_idx)
        cluster_list.sort(key=lambda x:len(x))
        index_select = []
        cluster_index = 0
        # print("Select cluster",len(set(hac_list)))
        while k > 0:
            if len(cluster_list[cluster_index]) > 0:
                index_select.append(cluster_list[cluster_index].pop(0)) 
                k -= 1
            if cluster_index < len(cluster_list) - 1:
                cluster_index += 1
            else:
                cluster_index = 0

        return index_select

    def query(self, k, current_round, js_divergence,param2,param3):

        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()

        if self.one_sample_step:
            self.one_sample_step = False
            self.emb_list = self.get_embeddings(unlabeled_data)
            self.HAC_list = AgglomerativeClustering(n_clusters=20, linkage = 'average').fit(self.emb_list)

        n = min(k*10,len(unlabeled_idxs))
        index = self.margin_data(n)
        index = self.round_robin(index, self.HAC_list.labels_, k)
        return index
