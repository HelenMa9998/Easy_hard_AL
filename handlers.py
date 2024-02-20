import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from monai import transforms
from seed import setup_seed
import torch
import albumentations as A
import bisect


setup_seed()
# get dataloader
    
class MSSEG_Handler_2d(Dataset):
    def __init__(self,image,label,mode="train"):
        self.image=np.array(image)
        self.label=np.array(label)
        self.classification_labels = []
        for i in label:
            sum_of_pixels = np.sum(i)
            if sum_of_pixels > 0:
                self.classification_labels.append(1)
            else:
                self.classification_labels.append(0)
        
        if mode=="train":
            self.transform = A.Compose([
                A.GaussianBlur(blur_limit=(5, 5), sigma_limit=0, always_apply=False, p=0.5),
                A.Flip(p=0.5),
                A.Rotate (limit=90, interpolation=1,always_apply=False, p=0.5),
                # A.Resize(width=256, height=256 ,p=1),
        ]) 
        else:
            self.transform = None
            
    def __len__(self):
        return len(self.classification_labels)
    def __getitem__(self,index): 
        img = self.image[index].astype(np.float32)
        seg_label = self.label[index].astype(np.int8)
        class_label = self.classification_labels[index]

        if self.transform!=None:
            img = self.transform(image=img)['image']

        img = torch.tensor(img)
        class_label = torch.tensor(class_label)
        seg_label = torch.tensor(seg_label)

        return img, class_label, seg_label, index


