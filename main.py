import argparse
import numpy as np
import torch
import pandas as pd
from pprint import pprint
import random
from data import Data
from utils import get_dataset, get_net, get_strategy, get_handler
from config import parse_args
from seed import setup_seed
# from visualization import visualiazation
import pdb
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from visualization import visualization


def main(param1,param2,param3):
    args = parse_args()
    pprint(vars(args))
    print()

    # fix random seed
    setup_seed()
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    os.environ['CUDA_LAUNCH_BLOCKINGs'] = '2'


    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # get dataset
    X_train, Y_train, X_val, Y_val, X_test, Y_test, handler = get_dataset(args.dataset_name,supervised=True)

    # get dataloader
    dataset = Data(X_train, Y_train, X_val, Y_val, X_test, Y_test, handler)


    # start experiment
    dataset.initialize_labels_random(args.n_init_labeled)
    # init_num, non_blank_idx = dataset.initialize_labels_K(train_num_slices_per_patient, args.k_init_labeled) # 或者在load dataset的时候就保证整除
    # dataset.initialize_labels(strategy_init, args.n_init_labeled)
    print(f"number of labeled pool: {args.n_init_labeled}")
    print(f"number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
    print(f"number of testing pool: {dataset.n_test}")
    print()

    # load prop network
    prop_net = get_net(args.dataset_name, device, prop=True) 
    # load AL strategy 这里有问题 train dataset是多个slice为一个样本
    strategy = get_strategy(param1)(dataset, prop_net) 
    # strategy = get_strategy(param1)(dataset, prop_net) # load strategy


    # Round 0 train 
    # 这里有另一种训练思路 即先训练一个模型（我理解是只有encoder和decoder?) 然后直接预测得到伪标签 然后使用伪标签和已标记数据进行标签传播 更新为标签
    print("Round 0")
    rd = 0
    # strategy.train(rd, args.training_name) #直接这里的train就是prop train
    accuracy = []
    recall = []
    precision = []
    f1_score = []
    area_under_curve = []
    specificity = []
    JS_divergency=[]

    size = []
    query_samples = []
    # Round 0 test 
    strategy.train(rd, args.training_name)
    test_preds,targets = strategy.predict(dataset.get_test_data())
    acc, rec, prec, f1, auc, spec = dataset.cal_test_acc(test_preds, targets)
    print(f"Round 0 testing accuracy: {acc, rec, prec, f1, auc, spec}")  # get model performance for test dataset
    accuracy.append(acc)
    recall.append(rec)
    precision.append(prec)
    f1_score.append(f1)
    area_under_curve.append(auc)
    specificity.append(spec)

    size.append(args.n_init_labeled)
    
    _, all_data = dataset.get_train_data()
    _, labeled_data = dataset.get_all_labeled_data()  # 假设的整个数据集
    all_embedding = strategy.get_embeddings(all_data).numpy()
    labeled_embedding = strategy.get_embeddings(labeled_data).numpy()
    js_div = dataset.compute_js_divergence(all_embedding, labeled_embedding)
    print(js_div)
    JS_divergency.append(js_div)
    # active learning selection

    for rd in range(1, args.n_round + 1):
        print(f"Round {rd}")
        # AL query

        query_idxs = strategy.query(args.n_query, rd, js_div,param2,param3)  # query_idxs为active learning请求标签的数据
        labeled_idxs, norm_train_loader = dataset.get_labeled_data()
        print("before",len(labeled_idxs))
        # update labels 这里也应该是slice为单位
        strategy.update(query_idxs)  # update training dataset and unlabeled dataset for active learning
        labeled_idxs, norm_train_loader = dataset.get_labeled_data()
        print("after",len(labeled_idxs))

        all_embedding = strategy.get_embeddings(all_data).numpy()
        _, labeled_data = dataset.get_all_labeled_data()  # 假设的整个数据集
        labeled_embedding = strategy.get_embeddings(labeled_data).numpy()
        js_div = dataset.compute_js_divergence(all_embedding, labeled_embedding)
        print(js_div)
        JS_divergency.append(js_div)
        query_samples.append(query_idxs)
        if js_div < float(param2):
            print(rd)
            print(query_samples)
            visualization(X_train, query_samples, param1,param2,param3)
            break
        strategy.train(rd, args.training_name)
        # prop_net.prop_train(prop_train_loader,prop_val_loader,rd)#只根据labeled data去学习encoder、key val encoder、decoder
        # pseudo_idxs, pseudo_label = prop_net.prop(prop_loader) 

        # calculate accuracy 这就是正常的 需要把计算key val以及cross attention去掉
        test_preds, targets = strategy.predict(dataset.get_test_data()) # get model prediction for test dataset
        acc, rec, prec, f1, auc, spec = dataset.cal_test_acc(test_preds, targets)
        print(f"Round {rd} testing accuracy: {acc, rec, prec, f1, auc, spec}")
        accuracy.append(acc)
        recall.append(rec)
        precision.append(prec)
        f1_score.append(f1)
        area_under_curve.append(auc)
        specificity.append(spec)
        # labeled_idxs, _ = dataset.get_labeled_data(eval_handler, pseudo_idxs=None)
        size.append(len(labeled_idxs))

    # save the result
    dataframe = pd.DataFrame(
        {'model': 'Unet', 'Method': args.strategy_name, 'Training dataset size': size, 'JS_divergency': JS_divergency, 'Accuracy': accuracy, 'Recall': recall, 'Precision': precision, 'F1_score': f1_score, 'Area_under_curve': area_under_curve, 'Specificity': specificity})
    dataframe.to_csv(f"./{param1}-1999-{param2}-{param3}.csv", index=False, sep=',')

experiment_parameters = [
    # param1 具体方法；param2 JS; param3 具体选择策略
    # normal; hard; [0-0.2]; [0-0.2]+[0.2-0.4]
    # {'param1': "CDALSampling", 'param2': None, 'param3': None},
    # {'param1': "LeastConfidence", 'param2': None, 'param3': None},
    # {'param1': "LeastConfidence", 'param2': '0.2', 'param3': "random+hard"},
    # {'param1': "EntropySampling", 'param2': '0.08', 'param3': "random+hard"},
    # {'param1': "EntropySampling", 'param2': '0.26', 'param3': "coreset+hard"},
    # {'param1': "EntropySampling", 'param2': '0.22', 'param3': "coreset+hard"},
    # {'param1': "EntropySampling", 'param2': '0.21', 'param3': "coreset+hard"},
    # {'param1': "EntropySampling", 'param2': '0.19', 'param3': "coreset+hard"},
    # {'param1': "EntropySampling", 'param2': '0.17', 'param3': "coreset+hard"},
    # {'param1': "EntropySampling", 'param2': '0.16', 'param3': "coreset+hard"},
    # {'param1': "EntropySampling", 'param2': '0.14', 'param3': "coreset+hard"},
    # {'param1': "EntropySampling", 'param2': '0.12', 'param3': "coreset+hard"},
    # {'param1': "EntropySampling", 'param2': '0.1', 'param3': "coreset+hard"},

    # {'param1': "LeastConfidence", 'param2': '0.11', 'param3': "kmedoid+hard"},
    {'param1': "LeastConfidence", 'param2': '0.08', 'param3': "random+hard"},
    # {'param1': "LeastConfidence", 'param2': '0.15', 'param3': "random+hard"},

    # {'param1': "EntropySampling", 'param2': '0.18', 'param3': "uncertainty_random+hard"},

    # {'param1': "MarginSampling", 'param2': '0.11', 'param3': "kmedoid+hard"},
    # {'param1': "MarginSampling", 'param2': '0.09', 'param3': "random+hard"},
    # {'param1': "MarginSampling", 'param2': '0.13', 'param3': "random+hard"},
    # {'param1': "MarginSampling", 'param2': '0.13', 'param3': "uncertainty_random+hard"},

    # {'param1': "EntropySamplingDropout", 'param2': None, 'param3': None},
    # {'param1': "EntropySamplingDropout", 'param2': '0.14', 'param3': "kmedoid+hard"},
    # {'param1': "LeastConfidence", 'param2': '0.19', 'param3': "kmedoid+hard"},
    # {'param1': "LeastConfidence", 'param2': '0.17', 'param3': "kmedoid+hard"},
    # {'param1': "LeastConfidence", 'param2': '0.16', 'param3': "kmedoid+hard"},
    # {'param1': "LeastConfidence", 'param2': '0.14', 'param3': "kmedoid+hard"},
    # {'param1': "LeastConfidence", 'param2': '0.13', 'param3': "kmedoid+hard"},
    # {'param1': "LeastConfidence", 'param2': '0.11', 'param3': "kmedoid+hard"},

    # {'param1': "LeastConfidence", 'param2': '0.08', 'param3': "uncertainty_random+hard"},
    # {'param1': "LeastConfidence", 'param2': '0.16', 'param3': "kmedoid+hard"},
    # {'param1': "LeastConfidence", 'param2': '0.14', 'param3': "kmedoid+hard"},
    # {'param1': "LeastConfidence", 'param2': '0.13', 'param3': "kmedoid+hard"},
    # {'param1': "LeastConfidence", 'param2': '0.11', 'param3': "kmedoid+hard"},

    # {'param1': "EntropySamplingDropout", 'param2': '0.12', 'param3': "random+hard"},
    # {'param1': "EntropySamplingDropout", 'param2': '0.15', 'param3': "random+hard"},
    # {'param1': "EntropySamplingDropout", 'param2': '0.11', 'param3': "uncertainty_random+hard"},


    # {'param1': "EntropySamplingDropout", 'param2': '0.2', 'param3': "random+hard"},
    # {'param1': "EntropySamplingDropout", 'param2': '0.18', 'param3': "random+hard"},
    # {'param1': "EntropySamplingDropout", 'param2': '0.17', 'param3': "random+hard"},
    # {'param1': "EntropySamplingDropout", 'param2': '0.15', 'param3': "random+hard"},
    # {'param1': "EntropySamplingDropout", 'param2': '0.13', 'param3': "random+hard"},
    # {'param1': "EntropySamplingDropout", 'param2': '0.12', 'param3': "random+hard"},
    # {'param1': "EntropySamplingDropout", 'param2': '0.1', 'param3': "random+hard"},
    # {'param1': "EntropySamplingDropout", 'param2': '0.08', 'param3': "random+hard"},

    # {'param1': "EntropySamplingDropout", 'param2': '0.2', 'param3': "random+hard"},
    # {'param1': "EntropySampling", 'param2': '0.12', 'param3': "kmedoid+hard"},
    # {'param1': "EntropySampling", 'param2': '0.18', 'param3': "kmedoid+hard"},
    # {'param1': "EntropySampling", 'param2': '0.2', 'param3': "kmedoid+hard"},

    # {'param1': "EntropySampling", 'param2': '0.16', 'param3': "uncertainty_random+hard"},
    # {'param1': "EntropySampling", 'param2': '0.18', 'param3': "uncertainty_random+hard"},

    # {'param1': "EntropySampling", 'param2': '0', 'param3': "random"},
    # {'param1': "EntropySamplingDropout", 'param2': '0', 'param3': "random"},
    # {'param1': "EntropySamplingDropout", 'param2': '0.12', 'param3': "hard"},
    # {'param1': "EntropySamplingDropout", 'param2': '0.13', 'param3': "hard"},
]

for params in experiment_parameters:
    main(params['param1'],params['param2'],params['param3'])