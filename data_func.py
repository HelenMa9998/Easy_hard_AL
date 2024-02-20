import nibabel as nib
import os
import numpy as np
import cv2
import torch
import SimpleITK as sitk

from seed import setup_seed

# provide the main method for data read and processing

setup_seed(42)
#both 2d and 3d 
def get_path(dir_name):
    centers = [f for f in sorted(os.listdir(dir_name)) if os.path.isdir(os.path.join(dir_name, f))]
    ps = []
    # print(centers)
    for i in centers:
    #     print(dir_name+i)
        patients = [f for f in sorted(os.listdir(dir_name+i)) if os.path.isdir(os.path.join(dir_name+i, f))]
        for j in patients:
            ps.append(i+"/"+j)
    return ps

def get_image(image_path,label=False):
    images = []
    num_slices_per_patient = []
    for i in range(len(image_path)):
        for j in range(len(image_path[i])):
            itk_img = sitk.ReadImage(image_path[i][j])
            image = sitk.GetArrayFromImage(itk_img)
            images.append(image)
            num_slices_per_patient.append(len(image))
    return np.array(images),num_slices_per_patient


# pre-processing
import matplotlib.pyplot as plt
def get_brain_area(images,masks,gts):
    brain_images = []
    brain_masks = []
    brain_area_masks = []
    for x in range(np.array(images).shape[0]):
        image = images[x]
        mask = masks[x]
        gt = gts[x]
#         print(mask.shape)
        target_indexs = np.where(mask == 1)
        w_maxs = np.max(np.array(target_indexs[0]))
        w_mins = np.min(np.array(target_indexs[0]))
        h_maxs = np.max(np.array(target_indexs[1]))
        h_mins = np.min(np.array(target_indexs[1]))
        d_maxs = np.max(np.array(target_indexs[2]))
        d_mins = np.min(np.array(target_indexs[2]))
#         print(w_maxs,w_mins,h_maxs,h_mins,d_maxs,d_mins)
        brain_image = image[w_mins:w_maxs, h_mins:h_maxs, d_mins:d_maxs]
        brain_mask = gt[w_mins:w_maxs, h_mins:h_maxs, d_mins:d_maxs]
        brain_area_mask = mask[w_mins:w_maxs, h_mins:h_maxs, d_mins:d_maxs]
#         print(brain_image.shape)
        brain_images.append(brain_image)
        brain_masks.append(brain_mask)
        brain_area_masks.append(brain_area_mask)
    return np.array(brain_images),np.array(brain_masks),np.array(brain_area_masks)

#get 2d slice from 3d patches
def get_2d_slice(images,labels,restrict=True,prop=False):
    slices = []
    num_slices = []
    patient_slices = []
    # index_list = []
    for x in range(np.array(images).shape[0]):
        for n_slice in range(np.array(images[x]).shape[0]):
            cbct_slice = images[x][n_slice,:,:]
            imgtlabel = labels[x][n_slice,:,:]
            prob = np.sum(imgtlabel)/(imgtlabel.shape[0]*imgtlabel.shape[1])
            if restrict==True:
                if prob==0:# 不加入
                    continue
            slices.append(np.array(cbct_slice))
            num_slices.append(np.array(images[x]).shape[0])
    # 将每个病人的图像分别放入一个列表中
    if prop == True: 
        start_idx = 0
        for num_slice in num_slices:
            end_idx = start_idx + num_slice
            patient_slices.append(list(range(start_idx, end_idx)))
            start_idx = end_idx

        # 将所有病人的图像列表合并
        slice_list = [slice for patient_slice in patient_slices for slice in patient_slice]
        # print(slice_list)
        # 构建index列表
        start_idx = 0
        for num_slice in num_slices:
            end_idx = start_idx + num_slice - 1
            # index_list.append((start_idx, end_idx))
            start_idx = end_idx + 1
    return np.array(slices)

# 假设所有的 2D 切片都存储在一个 Numpy 数组中，该数组的形状为 (num_slices, height, width)
# patient_ids 是一个与切片数量相同的 Numpy 数组，其中每个元素是一个唯一的病人标识符
def merge_slices_to_3D_image(slices, patient_ids):
    # 确定病人数量和每个病人的切片列表
    unique_patients = np.unique(patient_ids)
    num_patients = len(unique_patients)
    patient_slices = [[] for _ in range(num_patients)]

    # 将每个切片添加到相应的病人切片列表中
    for i, patient_id in enumerate(patient_ids):
        patient_index = np.where(unique_patients == patient_id)[0][0]
        patient_slices[patient_index].append(slices[i, :, :])

    # 将每个病人的切片合并为一个 3D 图像
    merged_image = []
    for i in range(num_patients):
        patient_image = np.stack(patient_slices[i], axis=0)
        merged_image.append(patient_image)

    return merged_image



# get 2d overlap patches from 2d slices
# def paint_border_overlap(full_imgs_all,patch_size=[128,128],stride=32):
#     patch_w,patch_h = patch_size
#     full_imgs_list = []
#     for full_imgs in full_imgs_all:
#         img_w = full_imgs.shape[0] #width of the image
#         img_h = full_imgs.shape[1]  #height of the image
#         leftover_w = (img_w-patch_w)%stride  #leftover on the w dim
#         leftover_h = (img_h-patch_h)%stride  #leftover on the h dim

#         if (leftover_w != 0):   #change dimension of img_w
#             tmp_full_imgs = np.zeros((img_w+(stride - leftover_w),img_h))
#             tmp_full_imgs[0:img_w,0:img_h] = full_imgs
#             full_imgs = tmp_full_imgs#in（144,512,512） out(160, 512, 512)

#         if (leftover_h != 0):  #change dimension of img_h
#             tmp_full_imgs = np.zeros((full_imgs.shape[0],img_h+(stride - leftover_h)))
#             tmp_full_imgs[0:full_imgs.shape[0],0:img_h] = full_imgs
#             full_imgs = tmp_full_imgs#out (160, 520, 512)
            
#         full_imgs_list.append(full_imgs)
#     return full_imgs_list


# def extract_ordered_overlap(full_imgs_all,label=None,patch_size=[128,128],stride=32,train=True):
#     patch_w,patch_h = patch_size
# #     w,h,d = patch_size
#     full_imgs_list = []
#     patches = []
#     label_patches = []
#     target_center = 0
#     non_target_center = 0
#     for x in range(np.array(full_imgs_all).shape[0]):
#         full_imgs = full_imgs_all[x]
#         img_w = full_imgs.shape[1] #width of the image
#         img_h = full_imgs.shape[0]  #height of the image
        
#         for j in range((img_h-patch_h)//stride+1):    
#             for i in range((img_w-patch_w)//stride+1):
#                     imgt = full_imgs[j*stride:j*stride+patch_h,i*stride:i*stride+patch_w]
#                     imgtlabel = label[x][j*stride:j*stride+patch_h,i*stride:i*stride+patch_w]
                    
#                     if train == True: 
#                         prob = np.sum(imgtlabel)/(imgtlabel.shape[0]*imgtlabel.shape[1])
#                         if prob==0:
#                             continue
#                         patches.append(imgt)
#                         label_patches.append(imgtlabel)
#                     else: 
#                         patches.append(imgt)
#                         label_patches.append(imgtlabel)
#     return patches,label_patches  #array with all the full_imgs divided in patches


#For 3D images recombine the 2D patches back to 3D iamges
# def paint_border_overlap_3d(full_imgs_all,patch_size=[64,64,64],stride=16):
#     patch_w,patch_h,patch_d = patch_size
#     full_imgs_list = []
#     for full_imgs in full_imgs_all:
#         img_w = full_imgs.shape[0] #width of the image
#         img_h = full_imgs.shape[1]  #height of the image
#         img_d = full_imgs.shape[2]  #depth of the image

#         leftover_w = (img_w-patch_w)%stride  #leftover on the w dim
#         leftover_h = (img_h-patch_h)%stride  #leftover on the h dim
#         leftover_d = (img_d-patch_d)%stride  #leftover on the h dim
    
#         if (leftover_w != 0):   #change dimension of img_w
#             tmp_full_imgs = np.zeros((img_w+(stride - leftover_w),img_h,img_d))
#             tmp_full_imgs[0:img_w,0:img_h,0:img_d] = full_imgs
#             full_imgs = tmp_full_imgs#in（144,512,512） out(160, 512, 512)
            
#         if (leftover_h != 0):  #change dimension of img_h
#             tmp_full_imgs = np.zeros((full_imgs.shape[0],img_h+(stride - leftover_h),img_d))
#             tmp_full_imgs[0:full_imgs.shape[0],0:img_h,0:img_d] = full_imgs
#             full_imgs = tmp_full_imgs#out (160, 520, 512)
            
#         if (leftover_d != 0):   #change dimension of img_w
#             tmp_full_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],img_d+(stride - leftover_d)))
#             tmp_full_imgs[0:full_imgs.shape[0],0:full_imgs.shape[1],0:img_d] = full_imgs
#             full_imgs = tmp_full_imgs
#         full_imgs_list.append(full_imgs)
#     return full_imgs_list


# def extract_ordered_overlap_3d(full_imgs_all,label=None,patch_size=[64,64,64],stride=16,train=True):
#     patch_w,patch_h,patch_d = patch_size
# #     w,h,d = patch_size
#     full_imgs_list = []
#     patches = []
#     for x in range(np.array(full_imgs_all).shape[0]):
#         full_imgs = full_imgs_all[x]
#         img_w = full_imgs.shape[0] #width of the image
#         img_h = full_imgs.shape[1]  #height of the image
#         img_d = full_imgs.shape[2]  #depth of the image
#         for i in range((img_w-patch_w)//stride+1):
#             for j in range((img_h-patch_h)//stride+1):
#                 for k in range((img_d-patch_d)//stride+1):
#                     imgt = full_imgs[i*stride:i*stride+patch_w,j*stride:j*stride+patch_h,k*stride:k*stride+patch_d]
#                     if train == True: 
#                         imgtlabel = label[x][i*stride:i*stride+patch_w,j*stride:j*stride+patch_h,k*stride:k*stride+patch_d]
#                         prob = np.sum(imgtlabel)/(imgtlabel.shape[0]*imgtlabel.shape[1]*imgtlabel.shape[2])
#                         if prob<0.005:# 不加入
#                             continue
#                         patches.append(imgt)
#                     else: 
#                         patches.append(imgt)
#     return patches  #array with all the full_imgs divided in patches


