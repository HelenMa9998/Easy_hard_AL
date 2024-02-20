import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import pandas as pde
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)
import time
import torch
import os
import tsnecuda
from tsnecuda import TSNE
# from data import get_Vessel
from handlers import MSSEG_Handler_2d
from utils import get_dataset
from seed import setup_seed

setup_seed(42)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# tsnecuda.test()


# def plot_embedding_2D(data, label, adversarial_samples, title):
#     x_min, x_max = np.min(data, 0), np.max(data, 0)
#     data = (data - x_min) / (x_max - x_min)
#     fig = plt.figure()
#     for i in range(data.shape[0]):
#         plt.text(data[i, 0], data[i, 1], str(label[i]),
#                  color=plt.cm.Set1(label[i]),
#                  fontdict={'weight': 'bold', 'size': 9})
#     plt.xticks([])
#     plt.yticks([])
#     plt.title(title)

def plot_embedding(X, query_sample, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    
    plt.figure()
    
    shown_images = np.array([[1., 1.]])  # just something big
    for i in range(X.shape[0]):
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 4e-3:
            # don't show points that are too close
            continue
        if np.max(dist) > 4e-3:
            # don't show points that are too far
            continue
        shown_images = np.r_[shown_images, [X[i]]]

    s1 = plt.scatter(X[:,0], X[:,1],s=0.5,c='cornflowerblue',marker='*')
    all_query_indices = np.concatenate(query_sample)  # 将所有查询样本索引合并
    all_query_samples = X[all_query_indices]  # 选取所有查询样本

    # 一次性绘制所有查询样本，所有样本使用同一种颜色
    s2 = plt.scatter(all_query_samples[:,0], all_query_samples[:,1], s=0.5, c='red', marker='o')
    plt.xticks([]), plt.yticks([])
    # plt.title(title)

    # for i in range(len(query_sample)):
    #     print(query_sample[i])
    #     query = X[query_sample[i]]#(100, 2)
    #     s2 = plt.scatter(query[:,0], query[:,1],s=0.5,c='r',marker='*')
    #     plt.xticks([]), plt.yticks([])
    #     plt.title(title)
    plt.savefig("picture/" + str(i) + ".png", dpi=1500)
    plt.legend((s1,s2),('All sample','Selected sample'))

    # plt.legend((s1,s2,s3,s4,s5,s6),('target <= 50','target <= 100','target <= 500','target <=1000','target > 1000','selected sample'))


def tSNE(X, query_sample, title):
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', perplexity=50, learning_rate=10, random_state=42)
    # tsne = TSNE(n_components=2, perplexity=15, learning_rate=10, random_state=42)
    # tsne = TSNE(n_components=2, perplexity=15, learning_rate=10)

    X_tsne = tsne.fit_transform(X)
    return X_tsne
    

def visualization(X_train, query_sample, param1,param2,param3):
    X_train = torch.tensor((X_train.reshape(X_train.shape[0], 240*240).astype('float32')) / 255.0)
    X_tsne = tSNE(X_train,query_sample,'x')

    plot_embedding(X_tsne, query_sample, "tSNE-"+param1+param2+param3)
    print("finished!!!")

# X_train, Y_train, X_val, Y_val, X_test, Y_test, handler, full_test_imgs_list, x_test_slice, test_brain_images, test_brain_masks = get_dataset('MSSEG',supervised=False)


# target_num = []
# for i in range(len(Y_train)):
#     target_num.append(np.sum(Y_train[i]))

# query_sample = [2288, 4627, 2536, 2659, 2271, 1724, 2771, 487, 3268, 3580, 4575, 5689, 4700, 6031, 3983, 56, 2875, 1107, 6487, 1182, 4738, 2104, 2407, 1841, 6990, 1672, 6613, 844, 1179, 4957, 4389, 2697, 6442, 1059, 3398, 835, 4898, 1464, 1222, 833, 7170, 915, 1340, 6143, 2008, 5953, 1217, 502, 1973, 2118, 4610, 5228, 2297, 725, 1450, 61, 6190, 2690, 3875, 6534, 1919, 187, 7043, 1583, 5063, 249, 6232, 6303, 2992, 377, 5293, 1102, 3627, 1223, 3496, 1913, 4210, 5879, 113, 4364, 3667, 949, 3676, 5128, 632, 6140, 1194, 19, 622, 664, 498, 1253, 2174, 3091, 3920, 65, 1477, 6113, 5743, 4770, 5246, 5078, 1360, 2069, 4009, 6343, 5084, 3732, 4468, 2954, 3369, 5981, 1446, 13, 4888, 4439, 224, 1183, 5470, 5518, 3517, 990, 5795, 384, 1756, 5249, 2369, 794, 3228, 3652, 4064, 2983, 846, 195, 3170, 4621, 182, 3309, 3980, 2215, 1404, 260, 1072, 4497, 1695, 5985, 2656, 5850, 3171, 5120, 4796, 3938, 877, 4206, 1041, 5755, 5993, 3888, 1371, 4165, 2526, 2593, 2936, 3416, 673, 1331, 2787, 681, 2335, 648, 769, 785, 2313, 474, 5705, 2392, 1191, 5564, 1366, 2772, 5859, 1087, 4424, 3817, 771, 3882, 154, 5086, 1884, 5258, 1893, 4919, 1651, 36, 1328, 4993, 5272, 2619, 5016, 1418, 711, 4267, 7, 1129, 222, 4873, 3066, 321, 2626, 2527, 97, 5500, 5282, 5135, 3451, 1114, 1626, 4754, 2818, 2827, 3204, 297, 2032, 4977, 3654, 171, 3916, 3638, 356, 2149, 3973, 1962, 1801, 3681, 2042, 2055, 4747, 4304, 5070, 2115, 1285, 483, 5039, 5196, 5023, 5319, 170, 2389, 908, 3816, 4923, 207, 2046, 415, 1316, 2937, 4447, 3556, 1936, 3656, 5136, 1030, 2504, 4668, 741, 5143, 4123, 3044, 581, 1710, 2959, 2717, 2840, 3593, 2700, 839, 141, 4730, 2906, 3011, 998, 2760, 2915, 4576, 2331, 302, 808, 569, 2419, 1795, 4319, 1852, 1620, 396, 3493, 3966, 1332, 3213, 4845, 2246, 2480, 886, 2572, 235, 3122, 1515, 2003, 3062, 3291, 3114, 164, 2761, 4524, 3482, 286, 513, 4696, 2077, 3903, 1538, 586, 1868, 3435, 1021, 824, 1391, 1335, 1980, 1204, 1897, 3391, 3871, 1208, 4179, 1415, 2667, 3775, 3863, 1971, 761, 955, 2921, 1259, 4108, 2095, 304, 2272, 434, 630, 3465, 2155, 192, 2181, 3279, 1952, 2233, 4084, 1712, 1150, 413, 927, 3560, 566, 3002, 758, 683, 48, 2683, 130, 1844, 2670, 875, 391, 1704, 1503, 335, 1849, 2301, 1346, 1141, 2289, 2412, 2197, 196, 1574, 2652, 2611, 3134, 1096, 1272, 3368, 3343, 2513, 1891, 407, 228, 3234, 1368, 552, 2509, 799, 2159, 2428, 2722, 3121, 1948, 3249, 723, 1061, 1965, 2627, 631, 2236, 521, 2576, 1401, 639, 1866, 2573, 2942, 101, 972, 1820, 2614, 1488, 811, 2098, 967, 578, 2113, 54, 1322, 1432, 1138, 422, 554, 594, 2373, 1075, 663, 43, 944, 1097, 39, 858, 1983, 740, 392, 282, 1348, 523, 146, 1954, 87, 375, 1907, 645, 231, 1384, 497, 893, 1525, 1146, 1035, 1396, 512, 1055, 1307, 1278, 1642, 214, 847, 98, 1164, 86, 80, 1239, 514, 731, 442, 1595, 602, 916, 1419, 72, 289, 636, 240, 270, 866, 203, 245, 827, 768, 401, 155, 284, 110, 122, 190]
# visualiazation(X_train, query_sample, target_num, 0, 'AdversarialAttact')
