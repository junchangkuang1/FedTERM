from http import client
import torch
import torchvision.transforms.functional as TF
# import matplotlib.pyplot as plt
import os
import copy
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomCrop
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import adjust_gamma as intensity_shift
import torch.nn as nn
# from scipy.ndimage import distance_transform_edt as distance
# from skimage import segmentation as skimage_seg
import numpy as np
# from dice_loss import dice_coeff
import random
import logging
from torch.nn import BCEWithLogitsLoss, BCELoss, CrossEntropyLoss
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
CE_Loss = BCELoss()
from dice_loss import dice_coeff
from medpy import metric
from PIL import Image

###############################################
#### CONSTANTS
###############################################

colors = ['r', 'g', 'b', 'c', 'k', 'y', 'm', 'c']


class TransformerAggregator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(TransformerAggregator, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.fc = nn.Linear(hidden_dim, input_dim)
        self.input_dim = input_dim

    def forward(self, local_model_output, global_model_output):
        # local_model_output: A (K, V)
        # global_model_output: B (Q)

        # 首先，将局部模型输出转换为合适的形状
        local_model_output = local_model_output.view(-1, 1, self.input_dim)  # (batch_size, 1, input_dim)
        global_model_output = global_model_output.view(-1, 1, self.input_dim)  # (batch_size, 1, input_dim)

        # 使用全局模型作为 Query，局部模型作为 Key 和 Value
        attn_output, attn_weights = self.attention(global_model_output, local_model_output, local_model_output)

        # 通过全连接层输出最终的聚合结果
        aggregated_output = self.fc(attn_output)
        return aggregated_output


def aggr_fed(CLIENTS, WEIGHTS_CL, nets, fed_name='global'):
    for param_tensor in nets[fed_name].state_dict():
        tmp = None

        for client, w in zip(CLIENTS, WEIGHTS_CL):
            if client != 'Interobs' and client != 'Lung422':
                if tmp == None:
                    tmp = copy.deepcopy(w * nets[client].state_dict()[param_tensor])
                else:
                    tmp += w * nets[client].state_dict()[param_tensor]
        nets[fed_name].state_dict()[param_tensor].data.copy_(tmp)
        del tmp




import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import copy

import copy
import torch





import torch
import torch.nn.functional as F



class BasicDataset(Dataset):
    def __init__(self, base_dir: str, split, train=False, transforms=None):
        print(split)
        self.transform = transforms  # using transform in torch!
        self.split = split
        self.image_list = []
        self._base_dir = base_dir
        self.train = train
        if train:
            with open(self._base_dir + '/{}_train.txt'.format(split), 'r') as f:
                self.image_list = f.readlines()

        else:
            with open(self._base_dir + '/{}_test.txt'.format(split), 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '') for item in self.image_list]

        print("{} has total {} samples".format(split, len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        image_name = self.image_list[idx]
        img_path = os.path.join(self._base_dir, self.split, 'images', image_name)
        mask_path = os.path.join(self._base_dir, self.split, 'masks', image_name)
        # print(f"Trying to open: {img_path}")
        image = np.array(Image.open(img_path).convert('L'))

        image = np.array(Image.open(img_path).convert('L'))
        mask = np.array(Image.open(mask_path).convert('L'))

        img = (image - np.min(image)) / (np.max(image) - np.min(image))
        mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
        mask[mask > 0] = 1
        mask[mask < 0] = 0
        sample = {'img': img, 'mask': mask, 'filename': img_path.split('\\')[-1]}
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
    def __init__(self, output_size, train=False):
        self.output_size = output_size
        self.train = train

    def __call__(self, sample):

        img, mask, filename = sample['img'], sample['mask'], sample['filename']
        if self.train:
            if random.random() > 0.5:
                img, mask = random_rot_flip(img, mask)
            elif random.random() > 0.5:
                img, mask = random_rotate(img, mask)
        x, y = img.shape
        # print('original shape: ',image.shape,label.shape)
        if x != self.output_size[0] or y != self.output_size[1]:
            img = zoom(img, (self.output_size[0] / x, self.output_size[1] / y), order=0)  # why not 3?
            mask = zoom(mask, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            # print(img, mask)
            # img = np.resize(img, (224,224))
            # mask = np.resize(mask, (224,224))
            # img = img.resize(self.output_size[0], self.output_size[1]) # why not 3?
            # mask = mask.resize(self.output_size[0], self.output_size[1])

        mask[mask >= 1] = 1

        img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        mask = torch.from_numpy(mask.astype(np.float32))
        sample = {'img': img, 'mask': mask, 'filename': filename}
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


###########################
## Test the network acc ###
###########################
def test(testloader, net, device, acc=None, loss=None):
    net.eval()
    t_loss, t_acc = 0, 0
    JC, ASSD, HD95 = 0, 0, 0
    # CE_Loss = BCEWithLogitsLoss()
    # Dice_Loss = DiceLoss(1)
    with torch.no_grad():
        for batch in testloader:
            image, mask_true = batch['img'], batch['mask']
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            sigmoid = nn.Sigmoid()
            # mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
            # print(mask_true.size())
            mask_true = mask_true.float()
            ###########################################
            # predict the mask
            mask_pred = net(image)
            mask_pred_norm = sigmoid(mask_pred.squeeze(1))
            # loss_ce = CE_Loss(mask_pred,mask_true.float())

            # loss_dice = dice_coeff(mask_pred,mask_true)
            # loss_total = 0.25*loss_ce + 0.75*loss_dice
            # t_loss += loss_total.item()

            # dice_loss += val_loss_dice
            # ce_loss += val_loss_ce
            # t_acc_network = dice_coeff(mask_true.type(torch.float), mask_pred).item()
            # t_acc += t_acc_network
            #######################################################
            mask_pred_1 = (mask_pred_norm > 0.5).float()
            if torch.sum(mask_pred_1) == 0 or torch.sum(mask_true) == 0:
                # print(torch.sum(mask_pred_1) ,torch.sum(mask_true), batch['filename'])
                # percentile = torch.quantile(mask_pred_norm, 0.0005)
                percentile = torch.max(mask_pred_norm)
                mask_pred_1 = (mask_pred_norm == percentile).float()
            t = mask_true.squeeze().cpu().numpy()
            p = mask_pred_1.squeeze().cpu().numpy()

            t_acc_network = metric.binary.dc(t, p)
            jc = metric.binary.jc(t, p)

            asd = metric.binary.asd(t, p)

            hd95 = metric.binary.hd95(t, p)

            t_acc += t_acc_network
            JC += jc
            ASSD += asd
            HD95 += hd95
    if acc is not None:
        acc.append(t_acc / len(testloader))
        # print('val_loss_ce: ',val_loss_ce / len(testloader),'val_loss_dice: ',val_loss_dice / len(testloader),'acc: ',t_acc / len(testloader) )

    if loss is not None:
        loss.append(t_loss / len(testloader))
    # del t_acc, t_loss

    return t_acc / len(testloader), JC / len(testloader), ASSD / len(testloader), HD95 / len(testloader)
    # print(t_acc / len(testloader))
    # print(JC / len(testloader))
    # return t_acc / len(testloader), JC / len(testloader)


import torch
import torch.nn.functional as F  # 需要导入 F，否则会报错




# CE_LOSS = nn.BCELoss()
def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def local_entropy(p, kernel_size=5):
    """
    计算局部熵（Local Entropy）。

    :param p: 伪标签 (batch, num_classes, H, W)
    :param kernel_size: 局部计算窗口大小
    :return: 局部熵图 (batch, 1, H, W)
    """
    p = torch.clamp(p, 1e-6, 1.0)  # 避免log(0)
    entropy = -torch.sum(p * torch.log(p), dim=1, keepdim=True)  # (batch, 1, H, W)

    # 使用均值池化来计算局部熵
    local_entropy = F.avg_pool2d(entropy, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    return local_entropy


def loss_local_entropy(p1, p2, kernel_size=5):
    """
    计算局部熵一致性损失。

    :param p1: 伪标签 1 (batch, num_classes, H, W)
    :param p2: 伪标签 2 (batch, num_classes, H, W)
    :return: 局部熵损失值
    """
    le1 = local_entropy(p1, kernel_size)
    le2 = local_entropy(p2, kernel_size)

    return torch.mean(torch.abs(le1 - le2))


def train_model(epoch, trainloader, optimizer_stu, device, net_stu, net_glb, ema_model=None,
                    acc=None, supervision_type='labeled', loss=None, learning_rate=None, iter_num=0,
                    cluster_centers_cross_client=None, sigma=0.1):

    net_stu.train()
    if ema_model is not None:
        ema_model.train()

    t_loss, t_acc = 0, 0
    max_iterations = 30000

    for i, batch in enumerate(trainloader):
        images = batch['img'].to(device=device, dtype=torch.float32)
        true_masks = batch['mask'].to(device=device, dtype=torch.int8)

        # Forward pass of student
        mask_pred = net_stu(images).squeeze(1)
        masks_pred = torch.sigmoid(mask_pred).float()

        if supervision_type == 'labeled':
            # Standard supervised loss
            loss_ce = CE_Loss(masks_pred, true_masks.float())
            loss_dice = (1 - dice_coeff(masks_pred, true_masks.float()))[0]
            loss_total = 0.25 * loss_ce + 0.75 * loss_dice


        else:

            features = net_stu.extract_features(images)

            B, C, H, W = features.shape

            features_flat = features.view(B, C, -1).permute(0, 2, 1).contiguous()

            if cluster_centers_cross_client is not None:

                cluster_centers = cluster_centers_cross_client.to(device)

                distances = torch.cdist(features_flat, cluster_centers)

                pseudo_probs = torch.softmax(-distances / sigma, dim=-1)

                pseudo_label_cpl = pseudo_probs.max(dim=-1)[0].view(B, H, W)

                diceloss = (1 - dice_coeff(pseudo_label_cpl, masks_pred))[0]

                loss_total = 0.25 * loss_local_entropy(pseudo_label_cpl, masks_pred) + 0.75 * diceloss


            else:

                loss_total = (1 - dice_coeff(masks_pred, masks_pred.detach()))[0]

            if ema_model is not None:
                update_ema_variables(net_stu, ema_model, 0.99, iter_num)
        # Update learning rate
        lr_ = learning_rate * (1.0 - iter_num / max_iterations) ** 0.9
        for param_group in optimizer_stu.param_groups:
            param_group['lr'] = lr_

        # Backpropagation
        iter_num += 1
        optimizer_stu.zero_grad()
        loss_total.backward()
        optimizer_stu.step()

        t_loss += loss_total.item()
        masks_pred_binary = (masks_pred.detach() > 0.5).float()
        t_acc += dice_coeff(masks_pred_binary, true_masks.float()).item()

    if acc is not None:
        acc.append(t_acc / len(trainloader))
    if loss is not None:
        loss.append(t_loss / len(trainloader))

    return t_acc / len(trainloader), t_loss / len(trainloader)


######################################
#def plot_graphs(num, CLIENTS, index, y_axis, title):
    #idx_clr = 0
   # plt.figure(num)
    #for client in CLIENTS:
        #plt.plot(index, y_axis[client], colors[idx_clr], label=client + title)
        #idx_clr += 1
    #plt.legend()
    #plt.show()


########################################
def train_fedmix(trainloader, net_stu, optimizer_stu, \
                 device, acc=None, loss=None, supervision_type='labeled', \
                 FedMix_network=1):
    net_stu.train()
    t_loss, t_acc = 0, 0
    # labeled_len = len(trainloader)
    # labeled_iter = iter(trainloader)
    for i, batch in enumerate(trainloader):
        imgs, masks, y_pl = batch['img'], batch['mask'], batch['y_pl']
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
            acc.append(t_acc / len(trainloader))
        except:
            acc.append(0.0)
    if loss is not None:
        try:
            loss.append(t_loss / len(trainloader))
        except:
            loss.append(0.0)


########################################
#### save model
#### input: PTH <saving path>
####      : epoch <identifier>
####      : nets [collection to save]
####      : acc_train : list of clients
#########################################


def save_model_4(PTH, dice, epoch, nets):
    torch.save(nets['global'], os.path.join(PTH, 'feddus_{}_net_{}.pth'.format(epoch, dice)))
    # torch.save(nets2, os.path.join(PTH , 'eam_ne_{}t.pth'.format(dice)))


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

