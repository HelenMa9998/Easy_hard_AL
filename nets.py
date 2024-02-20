import math
from turtle import shape
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
import torchvision
from collections import OrderedDict
from tqdm import tqdm
from seed import setup_seed
import pdb
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim


setup_seed()
# used for getting prediction results for data
def merge_slices_to_3D_image(slice_list, slices_per_patient):
    num_slices = len(slice_list)
    num_patients = len(slices_per_patient)
    num_rows, num_cols = slice_list[0].shape #病人的w h

    # Check if slices_per_patient sums up to total number of slices
    if sum(slices_per_patient) != num_slices:
        raise ValueError("Sum of slices_per_patient does not match total number of slices")
    # Create empty 3D image
    image = np.zeros((num_patients, num_rows, num_cols))
    # Merge slices for each patient
    current_idx = 0
    for i, num_slices in enumerate(slices_per_patient):
        # Concatenate slices for current patient
        image[i, :, :] = np.stack(slice_list[current_idx:current_idx+num_slices], axis=0)
        # Increment index counter
        current_idx += num_slices
    return image


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class dice_coefficient(nn.Module):
    def __init__(self, epsilon=0.0001):
        super(dice_coefficient, self).__init__()
        # smooth factor
        self.epsilon = epsilon

    def forward(self, targets, logits):
        batch_size = targets.shape[0]
        logits = (logits > 0.5).float()
        logits = logits.view(batch_size, -1).type(torch.FloatTensor)
        targets = targets.view(batch_size, -1).type(torch.FloatTensor)
        intersection = (logits * targets).sum(-1)
#         dice_score = 2. * (intersection + self.epsilon) / ((logits + targets).sum(-1) + self.epsilon)
        dice_score = (2. * intersection+ self.epsilon) / ((logits.sum(-1) + targets.sum(-1)) + self.epsilon)
        return torch.mean(dice_score)

# including different training method for active learning process (train acc=1, val loss, val acc, epoch)
class Net:
    def __init__(self, net, params, device):
        self.net = net
        self.params = params
        self.device = device

    def supervised_val_loss(self, data, val_data,rd):
        n_epoch = 100
        trigger = 0
        best = {'epoch': 1, 'loss': 10}
        train_loss=0
        validation_loss = 0
        train_dice=0
        val_dice=0
        self.clf = self.net().to(self.device)
        self.clf.train()
        if rd==0:
            self.clf = self.clf
        else:
            self.clf = torch.load('./result/model.pth')

        initial_lr = 0.0001
        optimizer = optim.Adam(self.clf.parameters(), lr=0.0001)
        
        # def update_learning_rate(optimizer, round_number, decay_rate):
        #     """ 更新学习率基于当前的主动学习轮次 """
        #     # 计算新的学习率
        #     if rd>7: 
        #         new_lr = initial_lr * (decay_rate ** (round_number-7))
        #     else: 
        #         new_lr = initial_lr
        #     # 设置新的学习率
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = new_lr
        #     print(f"Round: {round}, Learning Rate: {new_lr}")

        # update_learning_rate(optimizer, rd, decay_rate = 0.9 )
        criterion = nn.BCEWithLogitsLoss()
        # criterion = FocalLoss(alpha=1.6, gamma=2,logits=True)

        loader=DataLoader(data, **self.params['train_args'])
        val_loader=DataLoader(val_data, **self.params['val_args'])

        dice = dice_coefficient()
        for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
            for batch_idx, (x, y, seg, idxs) in enumerate(loader):#([8, 1, 240, 240])
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out,_ = self.clf(x)
                loss = criterion(out.squeeze().float(),y.float())
                loss.backward()
                optimizer.step()
                # train_dice += dice(y,sigmoid(out))
                # train_loss += loss#一个epoch的loss
            # print("\n epoch",epoch,"train loss: ",train_loss/(batch_idx+1),"train_dice: ",train_dice/(batch_idx+1))
            # clear loss and auc for training
            # train_loss=0
            # train_dice=0

            with torch.no_grad():
                self.clf.eval()
                for valbatch_idx, (valinputs, valtargets, seg, idxs) in enumerate(val_loader):
                    # valinputs, valtargets = valinputs.unsqueeze(1), valtargets.unsqueeze(1)
                    valinputs, valtargets = valinputs.to(self.device), valtargets.to(self.device)
                    valoutputs,_ = self.clf(valinputs)
                    validation_loss += criterion(valoutputs.squeeze().float(), valtargets.float())
            #         val_dice += dice(valtargets,sigmoid(valoutputs))
            # print(" epoch: ",epoch,"val loss: ",validation_loss/(valbatch_idx+1),"val_dice: ",val_dice/(valbatch_idx+1))
            
            trigger += 1
            # early stopping condition: if the acc not getting larger for over 10 epochs, stop
            if validation_loss / (valbatch_idx + 1) < best['loss']:
                trigger = 0
                best['epoch'] = epoch
                best['loss'] = validation_loss / (valbatch_idx + 1)
                # print(best['epoch'],best['loss'])
                torch.save(self.clf, './result/model.pth')
            # print("\n best performance at Epoch :{}, loss :{}".format(best['epoch'],best['loss']))
            validation_loss = 0
            # val_dice=0
            if trigger >= 5:
                break
        torch.cuda.empty_cache()
        
    
## restore to original dimensions
    def predict(self, data):
        self.clf = torch.load('./result/model.pth')
        self.clf.eval()
        preds = torch.zeros(len(data))
        labels= torch.zeros(len(data))
        loader = DataLoader(data, **self.params['test_args'])
        with torch.no_grad():
            for x, y, seg, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out,_ = self.clf(x)
                preds[idxs] = out.squeeze().cpu().float()
                labels[idxs] = y.cpu().float()
        return preds,labels
    
    # def predict(self, data, num_slices_per_patient):
    #     self.clf = torch.load('./result/model.pth')
    #     self.clf.eval()
    #     preds = []
    #     loader = DataLoader(data, **self.params['test_args'])
    #     with torch.no_grad(): 
    #         for x, y, idxs in loader:
    #             # x, y = x.unsqueeze(1), y.unsqueeze(1)
    #             x, y = x.to(self.device), y.to(self.device)
    #             out = self.clf(x, phase='test')
    #             outputs = out.data.cpu().numpy()
    #             preds.append(outputs)

    #     predictions = np.concatenate(preds, axis=0)#(40617, 1, 128, 128)
    #     # pred_patches = np.expand_dims(predictions,axis=1)#(40617, 1, 1, 128, 128)
    #     # pred_patches[pred_patches>=0.5] = 1
    #     # pred_patches[pred_patches<0.5] = 0
    #     # pred_imgs = recompone_overlap(pred_patches.squeeze(), full_test_imgs_list, x_test_slice, stride=96, image_num=8251)
    #     # pred_imgs_3d = recompone_overlap_3d(np.array(pred_imgs), test_brain_images, image_num=38)
    #     pred_imgs_3d = merge_slices_to_3D_image(np.array(predictions), num_slices_per_patient)
    #     pred_imgs_3d = np.array(pred_imgs_3d)
    #     return pred_imgs_3d


    def predict_prob(self, data):
        #         self.clf = torch.load('./model.pth')
        self.clf.eval()
        probs = torch.zeros([len(data), 1])
        loader = DataLoader(data, **self.params['test_args'])
        with torch.no_grad():
            for x, y, seg, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out,_ = self.clf(x)
                prob = torch.sigmoid(out)
                probs[idxs] = prob.cpu()
        return probs

    # Calculating 10 times probability for prediction, the mean used as uncertainty
    def predict_prob_dropout(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([len(data), 1])
        loader = DataLoader(data, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, seg, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out,_ = self.clf(x)
                    prob = torch.sigmoid(out)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        return probs

    # Used for Bayesian sampling
    def predict_prob_dropout_split(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([n_drop, len(data), 1])
        loader = DataLoader(data, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, seg, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out,_ = self.clf(x)
                    probs[i][idxs] += torch.sigmoid(out).cpu()
        return probs

    def get_embeddings(self, data):
        self.clf.eval()
        embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
        loader = DataLoader(data, **self.params['test_args'])
        with torch.no_grad():
            for x, y, seg, idxs in loader:
                # x, y = x.unsqueeze(1), y.unsqueeze(1)
                x, y = x.to(self.device), y.to(self.device)
                _, e1 = self.clf(x)
                embeddings[idxs] = e1.cpu().reshape(len(x),-1)
        return embeddings
    
    def predict_prob_embed(self, data, eval=True):
        loader = DataLoader(data, **self.params['test_args'])
        probs = torch.zeros([len(data), 1])
        embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])

        self.clf.eval()
        with torch.no_grad():
            for x, y, seg, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = torch.sigmoid(out)
                probs[idxs] = prob.cpu()
                embeddings[idxs] = e1.cpu()
        return probs, embeddings



class Dense_Net(nn.Module):
    def __init__(self):
        super(Dense_Net, self).__init__()
        # get layers of baseline model, loaded with some pre-trained weights on ImageNet
        self.feature_extractor = torchvision.models.densenet201(pretrained=True)
        self.feature_extractor.classifier = nn.Sequential()
        self.fc1 = nn.Linear(1920, 128, bias=True)
        self.fc2 = nn.Linear(128, 50, bias=True)
        self.fc3 = nn.Linear(50, 1, bias=True)
#        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, 1920)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x,e1
    
class Res_Net(nn.Module):
    def __init__(self):
        super(Res_Net, self).__init__()
        self.feature_extractor = torchvision.models.resnet18(pretrained=True)
        pretrained_first_conv = self.feature_extractor.conv1
        self.feature_extractor.conv1 = nn.Conv2d(
            1, 
            pretrained_first_conv.out_channels, 
            kernel_size=pretrained_first_conv.kernel_size, 
            stride=pretrained_first_conv.stride, 
            padding=pretrained_first_conv.padding, 
            bias=pretrained_first_conv.bias)
        self.feature_extractor.conv1.weight.data = pretrained_first_conv.weight.data.mean(dim=1, keepdim=True)

        self.feature_extractor.fc = nn.Sequential()
        
        self.fc1 = nn.Linear(512, 128, bias=True)
        self.fc2 = nn.Linear(128, 1, bias=True)

    def forward(self, x):
        e1 = self.feature_extractor(x)
        x = F.dropout(e1, training=self.training)
        x = self.fc1(e1)
        x = self.fc2(x)
        return x,e1
    
    def get_embedding_dim(self):
        return 512

