from http import client
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import os
import copy
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomCrop
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import adjust_gamma as intensity_shift
import torch.nn as nn
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import numpy as np
# from dice_loss import dice_coeff
import random
import logging
from torch.nn import BCEWithLogitsLoss,BCELoss,CrossEntropyLoss
# from monai.losses.focal_loss import FocalLoss
# from monai.losses.tversky import TverskyLoss

from os import listdir
from os.path import splitext
from pathlib import Path
import torch.optim as optim

from scipy import ndimage
from scipy.ndimage import zoom
import argparse
sigmoid = nn.Sigmoid()
CE_Loss =BCELoss()
from dice_loss import dice_coeff
import cleanlab
###############################################
#### CONSTANTS 
###############################################

colors = ['r', 'g', 'b', 'c', 'k', 'y','m', 'c']

def aggr_fed(CLIENTS, WEIGHTS_CL, nets, fed_name='global'):

    for param_tensor in nets[fed_name].state_dict():
        tmp= None

        for client, w in zip(CLIENTS, WEIGHTS_CL):
            if client != 'Interobs' and client != 'Lung422':
                if tmp == None:
                    tmp = copy.deepcopy(w*nets[client].state_dict()[param_tensor])
                else:
                    tmp += w*nets[client].state_dict()[param_tensor]
        nets[fed_name].state_dict()[param_tensor].data.copy_(tmp)
        del tmp

class cancer_v2(Dataset):
    def __init__(self, im_store, pl1_store, pl2_store):
        self.im_store = im_store
        self.pl1_store = pl1_store
        self.pl2_store = pl2_store
        
    def __len__(self):
        return len(self.im_store)
    
    def __getitem__(self, idx):
        x,y1,y2 = self.im_store[idx], self.pl1_store[idx], self.pl2_store[idx]
        sample = {'img': x, 'mask': y1,'y_pl':y2}
        return sample

class BasicDataset_feilei(Dataset):
    def __init__(self, base_dir: str, split, transforms = None):
        self.transform = transforms  # using transform in torch!
        self.split = split
        self.image_list = []
        self._base_dir = base_dir


        if split == 'ARGOS':
            with open(self._base_dir + '分类txt_2d/Argos_feibi.txt', 'r') as f:
                self.image_list = f.readlines()

        elif split == 'shantou':
            with open(self._base_dir+'分类txt_2d/shantou_feibi.txt', 'r') as f:
                self.image_list = f.readlines()

        elif split == 'rider':
            with open(self._base_dir+'分类txt_2d/rider_feibi.txt', 'r') as f:
                self.image_list = f.readlines()

        elif split == 'SYHuge':
            with open(self._base_dir+'分类txt_2d/Syhuge_feibi.txt', 'r') as f:
                self.image_list = f.readlines()
                # print("SYhuge",self.image_list)

        elif split == 'Interobs':
            with open(self._base_dir+'分类txt_2d/interobs_feibi.txt', 'r') as f:
                self.image_list = f.readlines()

        elif split == 'Lung422':
            with open(self._base_dir+'分类txt_2d/lung1_feibi.txt', 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n','') for item in self.image_list]

        print("{} has total {} samples".format(split,len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        image_name = self.image_list[idx]

        if self.split == 'ARGOS':
            path = self._base_dir + 'Argos/' + image_name  + ".npz"
            img = np.load(path)['img']
            mask = np.load(path)['mask']
            
        elif self.split == 'rider':
            path = self._base_dir + 'rider/' + image_name  
            img = np.load(path)['img']
            mask = np.load(path)['mask']
        
        elif self.split == 'shantou':
            path = self._base_dir + 'shantou/' + image_name  
            img = np.load(path)['img']
            mask = np.load(path)['mask']
    
        elif self.split == 'SYHuge':
            path = self._base_dir + 'SYhuge/' + image_name  
            img = np.load(path)['img']
            mask = np.load(path)['mask']

        elif self.split == 'Interobs':
            path = self._base_dir + 'interobs/' + image_name  
            img = np.load(path)['img']
            mask = np.load(path)['mask']

        elif self.split == 'Lung422':
            path = self._base_dir + 'Lung/' + image_name  
            img = np.load(path)['img']
            mask = np.load(path)['mask']

        sample = {'img': img, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)
        return sample

class BasicDataset_test(Dataset):
    def __init__(self, base_dir: str, split, transforms = None):
        self.transform = transforms  # using transform in torch!
        self.split = split
        self.image_list = []
        self._base_dir = base_dir


        if split == 'SX':
            with open(self._base_dir + 'txt_tvt/SX_test.txt', 'r') as f:
                self.image_list = f.readlines()

        elif split == 'JM':
            with open(self._base_dir+'txt_tvt/JM_test.txt', 'r') as f:
                self.image_list = f.readlines()

        elif split == 'rider':
            with open(self._base_dir+'txt_tvt/rider_test.txt', 'r') as f:
                self.image_list = f.readlines()
        elif split == 'MSD':
            with open(self._base_dir+'txt_tvt/MSD_test.txt', 'r') as f:
                self.image_list = f.readlines()
        elif split == 'LNDb':
            with open(self._base_dir+'txt_tvt/LNDb_test.txt', 'r') as f:
                self.image_list = f.readlines()
                # print("SYhuge",self.image_list)

        elif split == 'Interobs':
            with open(self._base_dir+'txt_tvt/interobs_external.txt', 'r') as f:
                self.image_list = f.readlines()

        elif split == 'Lung422':
            with open(self._base_dir+'txt_tvt/Lung.txt', 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n','') for item in self.image_list]

        print("{} has total {} samples".format(split,len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        image_name = self.image_list[idx]

        if self.split == 'ARGOS':
            path = self._base_dir + 'Argos/' + image_name  + ".npz"
            img = np.load(path)['img']
            mask = np.load(path)['mask']
            
        elif self.split == 'rider':
            path = self._base_dir + 'rider/' + image_name  
            img = np.load(path)['img']
            mask = np.load(path)['mask']
        
        elif self.split == 'shantou':
            path = self._base_dir + 'shantou/' + image_name  
            img = np.load(path)['img']
            mask = np.load(path)['mask']
    
        elif self.split == 'SYHuge':
            path = self._base_dir + 'SYhuge/' + image_name  
            img = np.load(path)['img']
            mask = np.load(path)['mask']

        elif self.split == 'Interobs':
            path = self._base_dir + 'interobs/' + image_name  
            img = np.load(path)['img']
            mask = np.load(path)['mask']

        elif self.split == 'Lung422':
            path = self._base_dir + 'Lung/' + image_name  
            img = np.load(path)['img']
            mask = np.load(path)['mask']

        sample = {'img': img, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)
        return sample

class BasicDataset(Dataset):
    def __init__(self, base_dir: str, split, train = False, transforms = None):
        print(split)
        self.transform = transforms  # using transform in torch!
        self.split = split
        self.image_list = []
        self._base_dir = base_dir
        self.train = train
        if train == 'train':
            if split == 'GD':
                with open(self._base_dir + 'txt_tvt/GD_train.txt', 'r') as f:
                    self.image_list = f.readlines()
            elif split == 'MSD':
                print(self._base_dir+'/txt_tvt/MSD_train.txt')
                with open(self._base_dir+'txt_tvt/MSD_train.txt', 'r') as f:
                    self.image_list = f.readlines()
            elif split == 'Rider':
                print(self._base_dir+'/txt_tvt/rider_train.txt')
                with open(self._base_dir+'txt_tvt/rider_train.txt', 'r') as f:
                    self.image_list = f.readlines()
            elif split == 'LNDb':
                print(self._base_dir+'/txt_tvt/rLNDb_train.txt')
                with open(self._base_dir+'txt_tvt/LNDb_train.txt', 'r') as f:
                    self.image_list = f.readlines()
            elif split == 'SX':
                with open(self._base_dir+'txt_tvt/SX_train.txt', 'r') as f:
                    self.image_list = f.readlines()
            elif split == 'JM':
                with open(self._base_dir+'txt_tvt/JM_train.txt', 'r') as f:
                    self.image_list = f.readlines()

        elif train == 'val':
            if split == 'GD':
                with open(self._base_dir + 'txt_tvt/GD_valid.txt', 'r') as f:
                    self.image_list = f.readlines()
            elif split == 'LNDb':
                print(self._base_dir+'/txt_tvt/LNDb_valid.txt')
                with open(self._base_dir+'txt_tvt/LNDb_valid.txt', 'r') as f:
                    self.image_list = f.readlines()
            elif split == 'SX':
                with open(self._base_dir+'txt_tvt/SX_valid.txt', 'r') as f:
                    self.image_list = f.readlines()

            elif split == 'Rider':
                with open(self._base_dir+'txt_tvt/rider_valid.txt', 'r') as f:
                    self.image_list = f.readlines()
            elif split == 'MSD':
                with open(self._base_dir + 'txt_tvt/MSD_valid.txt', 'r') as f:
                    self.image_list = f.readlines()
            elif split == 'JM':
                with open(self._base_dir+'txt_tvt/JM_valid.txt', 'r') as f:
                    self.image_list = f.readlines()

            elif split == 'Interobs':
                with open(self._base_dir+'txt_tvt/interobs_external.txt', 'r') as f:
                    self.image_list = f.readlines()

            elif split == 'Lung422':
                with open(self._base_dir+'txt_tvt/Lung422_test.txt', 'r') as f:
                    self.image_list = f.readlines()

        elif train == 'test':
            if split == 'GD':
                with open(self._base_dir + 'txt_tvt/GD_test.txt', 'r') as f:
                    self.image_list = f.readlines()
            elif split == 'LNDb':
                print(self._base_dir+'/txt_tvt/LNDb_test.txt')
                with open(self._base_dir+'txt_tvt/LNDb_test.txt', 'r') as f:
                    self.image_list = f.readlines()
            elif split == 'SX':
                with open(self._base_dir+'txt_tvt/SX_test.txt', 'r') as f:
                    self.image_list = f.readlines()

            elif split == 'Rider':
                with open(self._base_dir+'txt_tvt/rider_test.txt', 'r') as f:
                    self.image_list = f.readlines()
            elif split == 'MSD':
                with open(self._base_dir + 'txt_tvt/MSD_test.txt', 'r') as f:
                    self.image_list = f.readlines()
            elif split == 'JM':
                with open(self._base_dir+'txt_tvt/JM_test.txt', 'r') as f:
                    self.image_list = f.readlines()

            elif split == 'Interobs':
                with open(self._base_dir+'txt_tvt/interobs_external.txt', 'r') as f:
                    self.image_list = f.readlines()

            elif split == 'Lung422':
                with open(self._base_dir+'txt_tvt/Lung422_test.txt', 'r') as f:
                    self.image_list = f.readlines()

        self.image_list = [item.replace('\n','') for item in self.image_list]

        print("{} has total {} samples".format(split,len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        image_name = self.image_list[idx]
        path = os.path.join(self._base_dir , self.split , image_name)
        # if self.split == 'ARGOS':
        #     path = self._base_dir + 'ARGOS/' + image_name
            # img = np.load(path)['img']
            # mask = np.load(path)['mask']

        # elif self.split == 'rider':
        #     path = self._base_dir + 'rider/' + image_name
            # img = np.load(path)['img']
            # mask = np.load(path)['mask']
        
        # elif self.split == 'shantou':
        #     path = self._base_dir + 'shantou/' + image_name
            # img = np.load(path)['img']
            # mask = np.load(path)['mask']

        # elif self.split == 'SYHuge':
        #     path = self._base_dir + 'SYhuge/' + image_name
            # img = np.load(path)['img']
            # mask = np.load(path)['mask']
        
        # elif self.split == 'Interobs':
        #     path = self._base_dir + 'interobs/' + image_name
            # img = np.load(path)['img']
            # mask = np.load(path)['mask']

        # elif self.split == 'Lung422':
        #     path = self._base_dir + 'Lung422/' + image_name
            # img = np.load(path)['img']
            # mask = np.load(path)['mask']
        img = np.load(path)['img']
        mask = np.load(path)['mask']

        sample = {'img': img, 'mask': mask,'y_pl':mask}
        if self.transform:
            sample = self.transform(sample)
        return sample

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size, train = False):
        self.output_size = output_size
        self.train = train

    def __call__(self, sample):

        img,mask = sample['img'],sample['mask']
        if self.train:
            if random.random() > 0.5:
                img, mask = random_rot_flip(img, mask)
            elif random.random() > 0.5:
                img, mask = random_rotate(img, mask)
        x, y = img.shape
        # print('original shape: ',image.shape,label.shape)
        if x != self.output_size[0] or y != self.output_size[1]:
            img = zoom(img, (self.output_size[0] / x, self.output_size[1] / y), order = 0)  # why not 3?
            mask = zoom(mask, (self.output_size[0] / x, self.output_size[1] / y), order = 0)
        # print(image.shape,label.shape)
        mask[mask >= 1] = 1

        img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        mask = torch.from_numpy(mask.astype(np.float32))
        sample = {'img': img, 'mask': mask.long(), 'y_pl': mask.long()}
        return sample


 

############################################
#### copy federated model to client 
#### input: CLIENTS <list of client>
####      : nets <collection of dictionaries>
############################################
def copy_fed(CLIENTS, nets, fed_name='global'):
    for client in CLIENTS:
        if 'Interobs' not in client and 'Lung422' not in client:
            nets[client].load_state_dict(copy.deepcopy(nets[fed_name].state_dict()))

#############################################
### A helper function to randomly find bbox #
#############################################


def select_pl(nets_1, nets_2, device, trainloader, im_store, \
    pl1_store, pl2_store, \
    TH = 0.9, bbox=False):
    counter, dice_acc = 0,0 # create variable to store the num data and accuracy
    nets_1.eval()
    nets_2.eval()
    with torch.no_grad():
        # trainloader contains the actual label but is not used # 
        for batch in trainloader:
            imgs,masks = batch['img'],batch['mask']          
            imgs_cuda1, imgs_cuda2 = imgs.to(device), imgs.to(device)

            y_pred, y2_pred = nets_1(imgs_cuda1), nets_2(imgs_cuda2)
            y_pred, y2_pred = torch.sigmoid(y_pred), torch.sigmoid(y2_pred)
            y_pred, y2_pred = (y_pred > 0.5).float(), (y2_pred > 0.5).float()

            dice_net12  = dice_coeff(y2_pred, y_pred)
            dice_wrt_gt = dice_coeff(masks.type(torch.float).to(device), y_pred)
            print(" dice_net12 is :",  dice_net12.item())
            print(" dice_wrt_gt is :", dice_wrt_gt.item())

            if dice_net12 >= TH:
                dice_acc += dice_wrt_gt
                counter += 1
                im_store.append(imgs[0])
                pl1_store.append(y_pred[0].detach().cpu())
                pl2_store.append(y2_pred[0].detach().cpu())
        
    # return the counter per total length and dice acc for evaluation

    try:
        return (dice_acc/counter, counter/len(trainloader))
    except:
        return (0,0)

###########################
## Test the network acc ###
###########################
# def test(epoch,testloader, net, device,supervision_t, acc=None, loss=None):
#     net.eval()
#     t_loss, t_acc = 0,0
#
#     # CE_Loss = BCEWithLogitsLoss()
#     # Dice_Loss = DiceLoss(1)
#     confidence = []
#     with torch.no_grad():
#         for batch in testloader:
#             image,mask_true = batch['img'],batch['mask']
#             # move images and labels to correct device and type
#             image = image.to(device=device, dtype=torch.float32)
#             mask_true = mask_true.to(device=device, dtype=torch.long)
#             sigmoid = nn.Sigmoid()
#             mask_true = mask_true.float()
#             ###########################################
#             # predict the mask
#             mask_pred = net(image)
#             mask_pred = sigmoid(mask_pred.squeeze(1))
#             loss_ce = CE_Loss(mask_pred,mask_true.float())
#
#             loss_dice = dice_coeff(mask_pred,mask_true)
#             loss_total = 0.25*loss_ce + 0.75*loss_dice
#             t_loss += loss_total.item()
#
#             # dice_loss += val_loss_dice
#             # ce_loss += val_loss_ce
#             #######################################################
#             masks_pred_np = mask_pred.cpu().detach().numpy()
#             masks_pred_np_1 = masks_pred_np[masks_pred_np > 0.5]
#             if len(masks_pred_np_1) != 0:
#                 confidence_1 = np.mean(masks_pred_np_1)
#             else:
#                 confidence_1 = 0
#
#             ll = masks_pred_np[masks_pred_np < 0.5]
#             confidence_2 = 1 - np.mean(ll)
#
#             # print('1',confidence_1)
#             s = confidence_1 + confidence_2
#             confidence.append(s)
#
#             mask_pred = (mask_pred>0.5).float()
#             # 计算非零元素的总和
#
#             t_acc_network = dice_coeff(mask_true.type(torch.float), mask_pred).item()
#             # print('val_loss_ce: ',val_loss_ce,'val_loss_dice: ',val_loss_dice,'acc: ',t_acc_network )
#             t_acc += t_acc_network
#
#     if acc is not None:
#         acc.append(t_acc / len(testloader))
#         # print('val_loss_ce: ',val_loss_ce / len(testloader),'val_loss_dice: ',val_loss_dice / len(testloader),'acc: ',t_acc / len(testloader) )
#
#     if loss is not None:
#         loss.append(t_loss/ len(testloader))
#     del t_acc, t_loss
#
#     confidence_epoch = sum(confidence)/len(confidence)
#     if supervision_t =='labeled':
#         pass
#     else:
#         if epoch < 150:
#             confidence_epoch = confidence_epoch * 0.3
#         else:
#             confidence_epoch = confidence_epoch * 0.6
#     return confidence_epoch

def calculate_entropy(probabilities):
    return -2*torch.sum(probabilities * torch.log2(probabilities + 1e-10)+(1-probabilities)*torch.log2(1-probabilities+1e-10), dim=1)

def test(epoch,testloader, net, device,supervision_t, acc=None, loss=None):
    net.eval()
    t_loss, t_acc = 0,0

    # CE_Loss = BCEWithLogitsLoss()
    # Dice_Loss = DiceLoss(1)
    uncertainty = []
    with torch.no_grad():
        for batch in testloader:
            image,mask_true = batch['img'],batch['mask']
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            sigmoid = nn.Sigmoid()
            mask_true = mask_true.float()

            # predict the mask
            mask_pred = net(image)
            mask_pred = sigmoid(mask_pred.squeeze(1))

            entropies = calculate_entropy(mask_pred)  # 计算熵
            uncertainty.append(torch.mean(entropies).item())

            loss_ce = CE_Loss(mask_pred,mask_true.float())

            loss_dice = dice_coeff(mask_pred,mask_true)
            loss_total = 0.25*loss_ce + 0.75*loss_dice
            t_loss += loss_total.item()

            mask_pred = (mask_pred>0.5).float()
            # 计算非零元素的总和

            t_acc_network = dice_coeff(mask_true.type(torch.float), mask_pred).item()
            t_acc += t_acc_network



    confidence_epoch = sum(uncertainty)/len(uncertainty)
    loss_epoch = t_loss /len(testloader)
    if supervision_t =='labeled':
        pass
    else:
        if epoch < 150:
            confidence_epoch = confidence_epoch * 0.3
            loss_epoch = loss_epoch * 2
        else:
            confidence_epoch = confidence_epoch * 0.6
            loss_epoch = loss_epoch * 1.5

    if acc is not None:
        acc.append(t_acc / len(testloader))

    if loss is not None:
        loss.append(t_loss / len(testloader))
    del t_acc, t_loss

    return confidence_epoch,loss_epoch

'''
Training for every method
if FedST, we augment the image and use crossentropy
'''
# CE_LOSS = nn.BCELoss()
def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

import torch.nn.functional as F
def get_cam(model, input_tensor, target_class):
    # 将模型设为评估模式
    model.eval()

    with torch.no_grad():
        # class_score = output[:, target_class].unsqueeze(0)  # 选择目标类别的得分
        model.features = model.features.detach()  # 防止梯度传播

        # 获取最后卷积层的权重
        weights = model.outc.conv.weight[0]  # 假设 OutConv 的结构

        # 计算 CAM
        cam = F.relu(F.conv2d(model.features, weights.view(1, -1, 1, 1), bias=None))  # 得到 CAM
        cam = cam.squeeze().cpu().numpy()  # 转为 numpy 数组
        cam = cam - cam.min()  # 归一化
        cam = cam / cam.max()  # 归一化到 [0, 1]

        return cam

def enhance_segmentation_with_cam(segmentation_mask, cam, threshold=0.5):
    # 将 CAM 阈值化
    cam_thresholded = (cam > threshold).float()  # 阈值化

    # 与分割掩码结合
    enhanced_mask = segmentation_mask * cam_thresholded.squeeze()  # 仅保留高于阈值的区域
    return enhanced_mask

from PIL import Image
def train_model(epoch, trainloader, previous_nets, optimizer_stu, device, net_stu, ema_model = None, \
                      acc=None,supervision_type='labeled', \
                     loss = None,learning_rate=None,iter_num=0):
    net_stu.train()
    ema_model.train()
    t_loss, t_acc = 0,0    
    max_iterations = 30000

    for i,batch in enumerate(trainloader):
            images = batch['img']
            true_masks = batch['mask']
            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.int8)
            mask_pred = net_stu(images).squeeze(1)
            true_masks = true_masks.squeeze()

            masks_pred = sigmoid (mask_pred)
            masks_pred = masks_pred.float()

            if supervision_type == 'labeled':
                loss_ce = CE_Loss(masks_pred,true_masks.float())
                
                # loss_dice = Dice_Loss(masks_preds,true_masks )
                loss_dice = (1 - dice_coeff(masks_pred, true_masks.type(torch.float)))[0]
                loss_total = 0.25*loss_ce + 0.75*loss_dice

            else:
                noise = torch.clamp(torch.rand_like(images)*0.1, -0.2, 0.2)
                imgs_augmented = images + noise

                masks_pred = (masks_pred.detach() > 0.5).float()
                masks_augmented = sigmoid(ema_model(imgs_augmented).squeeze(1))

                loss_total = CE_Loss(masks_augmented, masks_pred)
                update_ema_variables(net_stu,ema_model,0.99,iter_num)

            lr_ = learning_rate * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer_stu.param_groups:
                param_group['lr'] = lr_
    
            iter_num += 1
           
            optimizer_stu.zero_grad()

            loss_total.backward()
            optimizer_stu.step()            
            
            t_loss += loss_total.item()

            # masks_pred = (masks_pred.detach() > 0.5).float()
            # pred_arr = masks_pred.squeeze().cpu()
            # pred_arr = np.array(pred_arr)
            # pred_arr = (pred_arr * 255).astype(np.uint8)
            # pred_arr = Image.fromarray(pred_arr)
            # if epoch > 4:
            #     pred_arr.save(r'G:\out\{}\pred\output_pred_{}_{}.jpg'.format(client, epoch, i), 'JPEG')
            #     image_array.save(r'G:\out\{}\img\output_image_{}_{}.jpg'.format(client,epoch,i), 'JPEG')
            #     mask_array.save(r'G:\out\{}\mask\output_mask_{}_{}.jpg'.format(client,epoch,i), 'JPEG')
            #     cam_array.save(r'G:\out\{}\cam\output_cam_{}_{}.jpg'.format(client,epoch,i), 'JPEG')
            t_acc_network = dice_coeff(masks_pred, true_masks.type(torch.float)).item()
            # print("dice is :",t_acc_network)
            t_acc += t_acc_network

    if acc is not None:
        try:
            acc.append(t_acc/len(trainloader))
        except:
            acc.append(0.0)
    if loss is not None:
        try:
            loss.append(t_loss/len(trainloader))
        except:
            loss.append(0.0)

    # loss_epoch = t_loss/len(trainloader)
    # if supervision_type == 'labeled':
    #     pass
    # else:
    #     if epoch < 150:
    #         loss_epoch = loss_epoch*2
    #     else:
    #         loss_epoch = loss_epoch * 1.5
    # return loss_epoch
    # print(sum(confidence)/len(confidence))
    # return sum(confidence)/len(confidence)

######################################
def plot_graphs(num, CLIENTS, index, y_axis, title):
    idx_clr = 0
    plt.figure(num)
    for client in CLIENTS:
        plt.plot(index, y_axis[client], colors[idx_clr], label=client+ title)
        idx_clr += 1
    plt.legend()
    plt.show()
########################################
def train_fedmix(trainloader, net_stu, optimizer_stu, \
                     device, acc=None, loss = None, supervision_type='labeled', \
                      FedMix_network=1):
    net_stu.train()
    t_loss, t_acc = 0,0 
    # labeled_len = len(trainloader)
    # labeled_iter = iter(trainloader)
    for i, batch in enumerate(trainloader):
        imgs, masks,y_pl = batch['img'], batch['mask'], batch['y_pl']
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer_stu.zero_grad()
        ###################################################
        l_ = 0
        ## get the prediction from the model of interest ##
        masks_stu = torch.sigmoid(net_stu(imgs))
        ### if supervision type is labeled, just train as normal with dice ###
        if supervision_type == 'labeled':
            l_stu = (1 - dice_coeff(masks_stu, masks.type(torch.float)))[0]
            l_ = l_stu
        else:
             if FedMix_network == 1:
                    masks_teach = y_pl.to(device)
             else:
                    masks_teach = masks.to(device)

             l_stu = (1 - dice_coeff(masks_stu, masks_teach.type(torch.float)))[0]
             l_ = l_stu
        #############################
        print("dice is :", l_.item())
        l_.backward()
        optimizer_stu.step()

        # for evaluation 
        t_loss += l_.item()
        masks_stu = (masks_stu.detach() > 0.5).float()
        t_acc_network = dice_coeff(masks_stu, masks.type(torch.float)).item()
        t_acc += t_acc_network
                
        
    if acc is not None:
        try:
            acc.append(t_acc/len(trainloader))
        except:
            acc.append(0.0)
    if loss is not None:
        try:
            loss.append(t_loss/len(trainloader))
        except:
            loss.append(0.0)

########################################
#### save model 
#### input: PTH <saving path>
####      : epoch <identifier>
####      : nets [collection to save]
####      : acc_train : list of clients 
#########################################
def save_model(PTH, epoch, nets, acc_train):
        p_global = PTH + 'avgglobal2'
        os.makedirs(p_global, exist_ok=True)
        torch.save(nets['global'], p_global + '/tvtmodel_' + str(epoch) +'.pth')

def save_model_3(PTH, epoch, nets, nets2):
        p_global = PTH + 'myglobal2023'
      
        torch.save(nets['global'], p_global + '/mytvtmodel_' + str(epoch) +'.pth')
        torch.save(nets2, p_global + '/mytvtemamodel_' + str(epoch) +'.pth')

def save_model_4(PTH, epoch, nets, nets2):
        p_global = PTH + 'myglobal20221210'
        torch.save(nets['global'], r"/root/autodl-tmp/my/model" + '/mytvtmodel_' + str(epoch) +'.pth')
        torch.save(nets2, r"/root/autodl-tmp/my/model" + '/mytvtemamodel_' + str(epoch) +'.pth')

def save_model_ll(PTH, epoch, nets, CLIENTS):
        for client in CLIENTS:
            p_global = PTH + 'llglobal/' + client
            os.makedirs(p_global, exist_ok=True)
            torch.save(nets[client], p_global + '/tvtmodel_' + str(epoch) + '.pth')

def save_model_centralize(PTH, epoch, nets):
        p_global = PTH + 'cenglobal2/'
        torch.save(nets, p_global + 'tvtmodel_' + str(epoch) + '.pth')

def sort_rows(matrix, num_rows):
    matrix_T = torch.transpose(matrix, 0, 1)
    sorted_T = torch.topk(matrix_T, num_rows)[0]
    return torch.transpose(sorted_T, 1, 0)

