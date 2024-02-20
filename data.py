from operator import index
import numpy as np
import torch
import glob
import os.path
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import cv2
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

from seed import setup_seed
from data_func import *
setup_seed()

import numpy as np

# 3D sice coefficient
def cal_subject_level_dice(prediction, target, class_num=2):# class_num是你分割的目标的类别个数
    eps = 1e-10
    empty_value = -1.0
    dscs = empty_value * np.ones((class_num), dtype=np.float32)
    for i in range(0, class_num):
        if i not in target and i not in prediction:
            continue
        target_per_class = np.where(target == i, 1, 0).astype(np.float32)
        prediction_per_class = np.where(prediction == i, 1, 0).astype(np.float32)

        tp = np.sum(prediction_per_class * target_per_class)
        fp = np.sum(prediction_per_class) - tp
        fn = np.sum(target_per_class) - tp
        dsc = 2 * tp / (2 * tp + fp + fn + eps)
        dscs[i] = dsc
    # dscs = np.where(dscs == -1.0, np.nan, dscs)
    subject_level_dice = np.nanmean(dscs[1:])
    return subject_level_dice

class dice_coefficient(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(dice_coefficient, self).__init__()
        # smooth factor
        self.epsilon = epsilon

    def forward(self, targets, logits):
        batch_size = 1
        logits[logits>=0.5] = 1
        logits[logits<0.5] = 0
        logits = logits.reshape(batch_size, -1)
        targets = targets.reshape(batch_size, -1)
        intersection = (logits * targets).sum(-1)
#         dice_score = 2. * intersection + self.epsilon / ((logits + targets).sum(-1) + self.epsilon)
#         dice_score = 2. * intersection / ((logits + targets).sum(-1) + self.epsilon)
        dice_score = (2. * intersection+ self.epsilon) / ((logits + targets).sum(-1) + self.epsilon)
#         print(dice_score)
        return dice_score
    
class Data:
    def __init__(self, X_train, Y_train, X_val, Y_val, X_test, Y_test, handler):
        self.X_train = X_train # used for maintaining original label
        self.Y_train = Y_train
        self.Y_train_pseudo = Y_train # used for pseudo training
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_test = X_test
        self.Y_test = Y_test 
        
        self.handler = handler

        self.n_pool = len(X_train)
        self.n_test = len(X_test)

        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        # self.unlabeled_idxs = np.zeros(self.n_pool, dtype=bool)

    def supervised_training_labels(self):
        # used for supervised learning baseline, put all data labeled
        tmp_idxs = np.arange(self.n_pool)
        self.labeled_idxs[tmp_idxs[:]] = True

    def initialize_labels_random(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True

    def initialize_labels_K(self, num_slices_per_patient, k):# 每个病人有多少slice
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        start_idx = 0
        non_blank_idx = []

        print("去除空白前X_train",self.X_train.shape) #(7750, 1, 240, 240)
        for i in range(len(num_slices_per_patient)):#([512, 512, 512, 512, 512, 256, 256, 256, 256, 256, 336, 336, 336, 336, 336])
            num_slices = num_slices_per_patient[i]
            num_full_segments = (num_slices // k)+1 # 每个病人多少初始化slice 25 
            last_segment_size = num_slices % k # 每个病人剩余多少slice 12 
            # print("num_full_segments",num_full_segments)
            selected_slices = [start_idx + k*j for j in range(num_full_segments)]#[0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480]
            print("selected_slices",selected_slices)
            start_idx += num_slices
            self.labeled_idxs[selected_slices] = True
            # print("selected_slices",selected_slices)
            # print("num_slices_per_patient",num_slices_per_patient)
            for j in range(len(selected_slices)-1): #[0, 30, 60, 90, 120, 150]   0 
                
                if np.sum(self.Y_train[selected_slices[j]])==0 and np.sum(self.Y_train[selected_slices[j+1]])!=0:
                    start_blank_idx = selected_slices[j]
                if np.sum(self.Y_train[selected_slices[j]])!=0 and np.sum(self.Y_train[selected_slices[j+1]])==0:
                    end_blank_idx = selected_slices[j+1]
            # print(start_blank_idx,end_blank_idx)
            non_blank_idx.extend(range(start_blank_idx, end_blank_idx+1))
        return len(np.arange(self.n_pool, dtype=int)[self.labeled_idxs]),non_blank_idx

    def delete_black_slices(self, index):
        self.X_train = self.X_train[index]
        self.Y_train = self.Y_train[index]
        self.labeled_idxs = self.labeled_idxs[index]
        self.n_pool = len(self.X_train)
        print("去除空白后X_train",self.X_train.shape)
        print(len(self.labeled_idxs))
        print("labeled",self.X_train[self.labeled_idxs].shape)
        print(len(self.labeled_idxs))
    
    # def initialize_labels_K(self, num_slices_per_patient, k):# 每个病人有多少slice
    #     self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
    #     start_idx = 0
    #     cumulative_counts = []
    #     cumulative_sum = 0

    #     for num_slices in num_slices_per_patient:
    #         num_full_segments = (num_slices // k) + 1
    #         last_segment_size = num_slices % k
    #         selected_slices = [start_idx + k*j for j in range(num_full_segments)]
    #         start_idx += num_slices
    #         # selected_slices.append(start_idx-1)  # Add the last slice
    #         self.labeled_idxs[selected_slices] = True #[0, 20, 40, 60, 80, 100, 120, 140]
    #         # print("selected_slices",selected_slices)
    #     # for count in num_slices_per_patient:
    #     #     cumulative_sum += count
    #     #     cumulative_counts.append(cumulative_sum)
    #     # print("num_slices_per_patient",cumulative_counts)
    #     #     if last_segment_size > 0:
    #     #         last_selected_slices = [i+k*num_full_segments for i in range(last_segment_size)]
    #     #         self.X_train = self.X_train[~last_selected_slices]
    #     #         self.Y_train = self.Y_train[~last_selected_slices]
    #     # print("initialize_labels_K data",self.X_train.shape,self.Y_train.shape)

    #     return len(np.arange(self.n_pool, dtype=int)[self.labeled_idxs]) # 8*50


    # def initialize_labels_K(self, interval): #病人slice 不能整除K
    #     tmp_idxs = np.zeros(self.n_pool, dtype=bool)
    #     for i in range(self.n_patients):
    #         start_idx = i * self.n_slices
    #         tmp_idxs[start_idx:start_idx+self.n_slices:interval] = True
    #     return len(tmp_idxs[tmp_idxs])

    # def get_labeled_data(self):
    #     # get labeled data for training
    #     labeled_idxs = np.arange(self.n_pool, dtype=int)[self.labeled_idxs]
    #     # print("labeled data", labeled_idxs.shape)
    #     # print("labeled_idxs ", labeled_idxs)
    #     return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs],mode="train")
    
    
    def get_labeled_data(self):
        # get labeled data for training
        labeled_idxs = np.arange(self.n_pool, dtype=int)[self.labeled_idxs]
        # print("labeled data", labeled_idxs.shape)
        # print("labeled_idxs ", labeled_idxs)
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs],mode="train")
    
    def get_embedding_data(self, index): #index是空白patch
        return self.handler(self.X_train[index], self.Y_train[index],mode="val")
    
    def get_filtered_data(self, X_train, Y_train):
        return self.handler(X_train, Y_train, mode="train")

    
    def get_data(self, pseudo_idxs, k, train_num_slices_per_patient): 
        # get labeled data for training
        # print(len(self.n_pool))
        # print(len(self.labeled_idxs))

        labeled_idxs = np.arange(self.n_pool, dtype=int)[self.labeled_idxs].tolist()
        print("normal",len(labeled_idxs))

        if pseudo_idxs != None:
            labeled_idxs.extend(pseudo_idxs) #把pseudo label加进去 进行采样
        labeled_idxs = np.array(labeled_idxs)
        # print(len(pseudo_idxs))
        print("pseudo",len(labeled_idxs))
        return labeled_idxs, self.handler(self.X_train, self.Y_train_pseudo, labeled_idxs, k, train_num_slices_per_patient)

    # def get_data(self, pseudo_idxs, k, train_num_slices_per_patient, handler): 
    #     # get labeled data for training
    #     labeled_idxs = np.arange(self.n_pool, dtype=int)[self.labeled_idxs]
    #     if pseudo_idxs != None:
    #         labeled_idxs = np.concatenate((np.arange(self.n_pool, dtype=int)[self.labeled_idxs], pseudo_idxs), axis=0)
    #     print(len(labeled_idxs))
    #     return labeled_idxs, handler(self.X_train, self.Y_train_pseudo, labeled_idxs, k, train_num_slices_per_patient)


    # used for pseudo label filter remove blank patches
    def delete_black_patch(self, index, preds):
        black_index = []
        for i in range(preds.shape[0]):#24537
            idx = preds[i]
            index[i]
            pred = (preds[i][1] > 0.5).int()
            if torch.sum(pred)==0:
                black_index.append(idx)
        return black_index

    # def get_unlabeled_data(self, index=None): #index是空白patch
    #     # get unlabeled data for active learning selection process
    #     unlabeled_idxs = np.arange(self.n_pool, dtype=int)[~self.labeled_idxs]#24537
    #     # print("unlabeled_idxs",unlabeled_idxs.shape)
    #     if index!=None:
    #         self.labeled_idxs[index] = True #5486
    #         unlabeled_idxs = np.arange(self.n_pool, dtype=int)[~self.labeled_idxs]#19051 19255
    #         self.labeled_idxs[index] = False
    #     return unlabeled_idxs
    
    def get_unlabeled_data(self, rd=None, index=None): #index是空白patch
        # get unlabeled data for active learning selection process
        unlabeled_idxs = np.arange(self.n_pool, dtype=int)[~self.labeled_idxs]#24537
        # print("unlabeled_idxs",unlabeled_idxs.shape)
        if index!=None:
            self.labeled_idxs[index] = True #5486
            unlabeled_idxs = np.arange(self.n_pool, dtype=int)[~self.labeled_idxs]#19051 19255
            self.labeled_idxs[index] = False
        # if rd ==8: 
            # print("get_unlabeled_data_x, get_unlabeled_data_y", self.X_train[unlabeled_idxs].shape, self.Y_train[unlabeled_idxs].shape)
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs],mode="val")

    # def compute_js_divergence(data1, data2, bins=30):
    #     """
    #     计算两个数据集之间的JS散度。
    #     :param data1: 第一个数据集（numpy数组）。
    #     :param data2: 第二个数据集（numpy数组）。
    #     :param bins: 构建直方图时使用的bins数量。
    #     :return: JS散度值。
    #     """
    #     # 计算直方图
    #     hist1, bin_edges = np.histogram(data1, bins=bins, density=True)
    #     hist2, _ = np.histogram(data2, bins=bin_edges, density=True)

    #     # 计算JS散度
    #     js_divergence = jensenshannon(hist1, hist2)
        
    #     return js_divergence

    def compute_js_divergence(self, data1, data2, bins=30):
        num_features = data1.shape[1]
        js_divergences = np.zeros(num_features)

        for i in range(num_features):
            # 计算每个特征的直方图
            hist1, _ = np.histogram(data1[:, i], bins=bins, density=True)
            hist2, _ = np.histogram(data2[:, i], bins=bins, density=True)

            # 计算JS散度
            js_divergences[i] = jensenshannon(hist1, hist2)

        # 返回所有特征的JS散度的平均值
        
        return np.mean(js_divergences)


    
    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train, mode="val")
    
    def get_all_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool, dtype=int)[self.labeled_idxs]
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs],mode="val")
    
    def get_cdal_labeled_data(self):
        # get labeled data for training
        labeled_idxs = np.arange(self.n_pool, dtype=int)[self.labeled_idxs]
        # print("labeled data", labeled_idxs.shape)
        # print("labeled_idxs ", labeled_idxs)
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs],mode="val")
        
    def get_val_data(self):
        # get validation dataset if exist
        return self.handler(self.X_val, self.Y_val,mode="val")

    def get_test_data(self):
        # get test dataset if exist
        return self.handler(self.X_test, self.Y_test,mode="val")

    # def cal_test_acc(self, logits, targets):
    #     # calculate accuracy for test dataset
    #     dscs = []
    #     for prediction, target in zip(logits, targets):
    #         dsc = cal_subject_level_dice(prediction, target, class_num=2)
    #         dscs.append(dsc)
    #         dice = np.mean(dscs)
    #     return dice

    def cal_test_acc(self, logits, targets):
        preds = (logits > 0.5).type(torch.int)  # 假设 threshold 是你设定的阈值

        # 确保 targets、preds 和 probs 都在 CPU 上并转换为 NumPy 数组
        targets_np = targets.cpu().numpy()
        preds_np = preds.cpu().numpy() # 类别
        probs_np = logits.cpu().numpy() # 概率

        # 计算各项指标
        acc = accuracy_score(targets_np, preds_np)
        recall = recall_score(targets_np, preds_np, average='weighted')
        precision = precision_score(targets_np, preds_np, average='weighted')
        f1 = f1_score(targets_np, preds_np, average='weighted')
        auc = roc_auc_score(targets_np, probs_np)  # 计算 AUC 分数

        # 计算特异性
        tn, fp, _, _ = confusion_matrix(targets_np, preds_np).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # 返回计算的指标
        return acc, recall, precision, f1, auc, specificity
    

    def add_labeled_data(self, data, label):
        # used for generated adversarial image expansion. Adding generated adversarial image with label to training dataset
        data = torch.reshape(data, (len(data),128,128))
        # data = torch.unsqueeze(data, 1)
        self.X_train = torch.tensor(self.X_train)#([25537, 128, 128])
        self.Y_train = torch.tensor(self.Y_train)
        self.X_train = torch.cat((self.X_train, data), 0)#([26037, 128, 128])
        self.Y_train = torch.cat((self.Y_train, label), 0)
        # print("labeled_idxs",self.labeled_idxs.shape)
        array = np.ones(len(data),dtype=bool)
        self.labeled_idxs = np.append(self.labeled_idxs, array)
        self.n_pool += len(data)
        return np.array(self.X_train)

    def get_label(self, idx):
        # Get the real label (share lable) for adversarial samples
        self.Y_train = np.array(self.Y_train)
        label = torch.tensor(self.Y_train[idx])
        return label

    def cal_target(self):
        target_num = []
        for i in range(len(self.Y_train)):
            target_num.append(np.sum(self.Y_train[i]))
        return target_num

def get_images(folders_name,train=False):
    image_path = r'../Task09_Spleen/imagesTr'
    mask_path = r'../Task09_Spleen/labelsTr'
    
    images = []
    masks = []
    num_slices_per_patient = []

    for fld_name in folders_name:
        path_img_flair = os.path.join(image_path, fld_name)
        #path_img_t1 = os.path.join(image_path, fld_name, fld_name + '_t1.nii.gz')
#         path_img_t1ce = os.path.join(image_path, fld_name, fld_name + '_t1ce.nii.gz')
        #path_img_t2 = os.path.join(image_path, fld_name, fld_name + '_t2.nii.gz')
        path_label = os.path.join(mask_path, fld_name)

        img_flair = sitk.ReadImage(path_img_flair)
        img_flair = sitk.GetArrayFromImage(img_flair)

        #img_t1 = sitk.ReadImage(path_img_t1)
        #img_t1 = sitk.GetArrayFromImage(img_t1)

#         img_t1ce = sitk.ReadImage(path_img_t1ce) #(155, 240, 240)
#         img_t1ce = sitk.GetArrayFromImage(img_t1ce)

        #img_t2 = sitk.ReadImage(path_img_t2)
        #img_t2 = sitk.GetArrayFromImage(img_t2)

        label = sitk.ReadImage(path_label)
        label = sitk.GetArrayFromImage(label)
        num_slices_per_patient.append(img_flair.shape[0])

        # label[(label >= 2)] = 1

        for index in range(0,img_flair.shape[0]):
            #img_flair_ = img_flair[index]
            #img_flair_ = np.expand_dims(img_flair_,axis=0)

            #img_t1_ = img_t1[index]
            #img_t1_ = np.expand_dims(img_t1_, axis=0)

            img_flair_ = img_flair[index]
            # img_flair_ = np.expand_dims(img_flair_, axis=0)

            #img_t2_ = img_t2[index]
            #img_t2_ = np.expand_dims(img_t2_, axis=0)
            label_ = label[index]
            # prob = np.sum(label_)
            # if train == True: 
            #     if prob==0:# 不加入
            #         continue
            
            
            #label_ = np.expand_dims(label_, axis=0)
            images.append(img_flair_)
            masks.append(label_)
        
    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.uint8)
    return images, masks, num_slices_per_patient

def get_MSSEG(handler,supervised = False):
    # train_ratio = 0.7
    # val_ratio = 0.15
    # test_ratio = 0.15
    # image_path = r'../Task09_Spleen/imagesTr'
    # # 遍历数据集路径下的所有文件
    # folders_name = os.listdir(image_path)
    # random.shuffle(folders_name)

    # # 计算数据集拆分的索引
    # num_train = int(len(folders_name) * train_ratio)
    # num_val = int(len(folders_name) * val_ratio)
    # num_test = len(folders_name) - num_train - num_val

    # # print(num_train)
    # train_path = folders_name[:num_train]
    # val_path = folders_name[num_train:num_train+num_val]
    # test_path = folders_name[num_train+num_val:num_train+num_val+num_test]
    # print(len(folders_name))
    # print("train",len(train_path))
    # print("val",len(val_path))
    # print("test",len(test_path))

    # x_train, y_train, train_num_slices_per_patient = get_images(train_path,train = False)
    # x_val, y_val, val_num_slices_per_patient = get_images(val_path)
    # x_test, y_test, test_num_slices_per_patient = get_images(test_path)
    # print(train_num_slices_per_patient)


    # x_train = np.array(x_train)
    # x_train = np.expand_dims(x_train, axis=1)
    # y_train = np.array(y_train)

    # x_val = np.array(x_val)
    # x_val = np.expand_dims(x_val, axis=1)
    # y_val = np.array(y_val)

    # x_test = np.array(x_test)
    # x_test = np.expand_dims(x_test, axis=1)
    # y_test = np.array(y_test)


    # np.save('../Task09_Spleen/train_num_slices_per_patient_2024.npy', train_num_slices_per_patient)
    # np.save('../Task09_Spleen/val_num_slices_per_patient_2024.npy', val_num_slices_per_patient)
    # np.save('../Task09_Spleen/test_num_slices_per_patient_2024.npy', test_num_slices_per_patient)


    # np.save('../Task09_Spleen/train_image_2024.npy', x_train)
    # np.save('../Task09_Spleen/train_label_2024.npy', y_train)
    # np.save('../Task09_Spleen/val_image_2024.npy', x_val)
    # np.save('../Task09_Spleen/val_label_2024.npy', y_val)
    # np.save('../Task09_Spleen/test_image_2024.npy', x_test)
    # np.save('../Task09_Spleen/test_label_2024.npy', y_test)



    # x_train = np.load('../Task09_Spleen/train_image.npy')
    # y_train = np.load('../Task09_Spleen/train_label.npy')
    # x_val = np.load('../Task09_Spleen/val_image.npy')
    # y_val = np.load('../Task09_Spleen/val_label.npy')
    # x_test = np.load('../Task09_Spleen/test_image.npy')
    # y_test = np.load('../Task09_Spleen/test_label.npy')

    # val_num_slices_per_patient = np.load('../Task09_Spleen/val_num_slices_per_patient.npy')

    # train_num_slices_per_patient = np.load('../Task09_Spleen/train_num_slices_per_patient.npy')
    # test_num_slices_per_patient = np.load('../Task09_Spleen/test_num_slices_per_patient.npy')

    # x_train = np.load('../BraTS2019/train_image_2024.npy')#(28055, 1, 240, 240)
    # y_train = np.load('../BraTS2019/train_label_2024.npy')
    # x_val = np.load('../BraTS2019/val_image_2024.npy')#(5890, 1, 240, 240)
    # y_val = np.load('../BraTS2019/val_label_2024.npy')
    # x_test = np.load('../BraTS2019/test_image_2024.npy')#(6200, 1, 240, 240)
    # y_test = np.load('../BraTS2019/test_label_2024.npy')

    # val_num_slices_per_patient = np.load('../BraTS2019/val_num_slices_per_patient_2024.npy')


    # train_num_slices_per_patient = np.load('../BraTS2019/train_num_slices_per_patient_2024.npy')
    # test_num_slices_per_patient = np.load('../BraTS2019/test_num_slices_per_patient_2024.npy')

    x_train = np.load('/home/siteng/hard_sample_AL/BraTS2019/train_image.npy')#(28055, 1, 240, 240)
    y_train = np.load('/home/siteng/hard_sample_AL/BraTS2019/train_label.npy')
    x_val = np.load('/home/siteng/hard_sample_AL/BraTS2019/val_image.npy')#(5890, 1, 240, 240)
    y_val = np.load('/home/siteng/hard_sample_AL/BraTS2019/val_label.npy')
    x_test = np.load('/home/siteng/hard_sample_AL/BraTS2019/test_image.npy')#(6200, 1, 240, 240)
    y_test = np.load('/home/siteng/hard_sample_AL/BraTS2019/test_label.npy')

    print(x_train.shape,y_train.shape)
    print(x_val.shape,y_val.shape)
    print(x_test.shape,y_test.shape)

    return x_train, y_train, x_val, y_val, x_test, y_test, handler