from http import client
import torch
from scipy.spatial.distance import cdist
from sklearn.metrics import jaccard_score
import os
import torchvision.transforms as transforms
from PIL import Image
import copy
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomCrop
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
# from dice_loss import dice_coeff
import random
import logging
from torch.nn import BCEWithLogitsLoss, BCELoss, CrossEntropyLoss
from medpy import metric


from scipy import ndimage
from scipy.ndimage import zoom
import argparse
from tqdm import tqdm
sigmoid = nn.Sigmoid()
CE_Loss = BCELoss()
from dice_loss import dice_coeff

###############################################
#### CONSTANTS
###############################################


def model_dist(w_1, w_2):
    dist_total = torch.zeros(1).float()
    w_1_dict = w_1.state_dict()  # 确保是状态字典
    w_2_dict = w_2.state_dict()  # 确保是状态字典

    for key in w_1_dict:
        dist = torch.norm(w_1_dict[key].cpu() - w_2_dict[key].cpu())
        dist_total += dist.cpu()

    return dist_total.cpu().item()

def aggr_fed(client_meta, WEIGHTS_CL, nets, fed_name='global'):

    for param_tensor in nets[fed_name].state_dict():
        tmp = None

        for client, w in zip(client_meta, WEIGHTS_CL):
            if client != 'Interobs' and client != 'Lung422':
                if tmp == None:
                    tmp = copy.deepcopy(w * nets[client].state_dict()[param_tensor])
                else:
                    tmp += w * nets[client].state_dict()[param_tensor]
        nets[fed_name].state_dict()[param_tensor].data.copy_(tmp)
        del tmp



class BasicDataset(Dataset):
    def __init__(self, base_dir: str, split, train=False, transforms=None):
        print(split)
        self.transform = transforms  # using transform in torch!
        self.split = split
        self.image_list = []
        self._base_dir = base_dir
        self.train = train
        if train == 'train':
            with open(self._base_dir + '/{}_train.txt'.format(split), 'r') as f:
                self.image_list = f.readlines()
        elif train == 'val':
            with open(self._base_dir + '/{}_val.txt'.format(split), 'r') as f:
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
        image = np.array(Image.open(img_path).convert('L'))
        mask = np.array(Image.open(mask_path).convert('L'))

        img = (image - np.min(image)) / (np.max(image)-np.min(image))
        mask = (mask - np.min(mask)) / (np.max(mask)-np.min(mask))
        mask[mask > 0] = 1
        mask[mask < 0] = 0
        sample = {'img': img, 'mask': mask,'filename':img_path.split('\\')[-1]}
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

        img, mask = sample['img'], sample['mask']
        if self.train:
            if random.random() > 0.5:
                img, mask = random_rot_flip(img, mask)
            elif random.random() > 0.5:
                img, mask = random_rotate(img, mask)
        x, y = img.shape
        # print('original shape: ',image.shape,label.shape)
        if x != self.output_size[0] or y != self.output_size[1]:
            # Final enforce resize regardless of above logic
            img = zoom(img, (self.output_size[0] / img.shape[0], self.output_size[1] / img.shape[1]), order=1)
            mask = zoom(mask, (self.output_size[0] / mask.shape[0], self.output_size[1] / mask.shape[1]), order=0)

        # print(image.shape,label.shape)
        mask[mask >= 1] = 1

        img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        mask = torch.from_numpy(mask.astype(np.float32))
        sample = {'img': img, 'mask': mask}
        return sample

def copy_fed(CLIENTS, nets, fed_name='global'):
    for client in CLIENTS:
        if 'Interobs' not in client and 'Lung422' not in client:
            nets[client].load_state_dict(copy.deepcopy(nets[fed_name].state_dict()))


def test(epoch, testloader, net, device, supervision_t, acc=None, loss=None):
    net.eval()
    t_loss, t_acc = 0, 0
    JC, ASSD, HD95 = 0, 0, 0
    with torch.no_grad():
        for batch in testloader:
            image, mask_true = batch['img'], batch['mask']
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            sigmoid = nn.Sigmoid()
            mask_true = mask_true.float()

            # predict the mask
            mask_pred = net(image)
            mask_pred_norm = sigmoid(mask_pred.squeeze(1))

            loss_ce = CE_Loss(mask_pred_norm, mask_true.float())

            loss_dice = dice_coeff(mask_pred_norm, mask_true)

            loss_total = 0.25 * loss_ce + 0.75 * loss_dice
            t_loss += loss_total.item()

            mask_pred_1 = (mask_pred_norm > 0.5).float()
            if torch.sum(mask_pred_1) == 0:
                # percentile = torch.quantile(mask_pred_norm, 0.0005)
                percentile = torch.max(mask_pred_norm)
                mask_pred_1 = (mask_pred_norm == percentile).float()
            # 计算非零元素的总和
            t = mask_true.cpu().numpy()
            p = mask_pred_1.cpu().numpy()

            # t_acc_network = dice_coeff(mask_true.type(torch.float), mask_pred).item()
            t_acc_network = metric.binary.dc(t, p)
            jc = metric.binary.jc(t, p)

            asd = metric.binary.asd(t, p)

            hd95 = metric.binary.hd95(t, p)

            t_acc += t_acc_network
            JC += jc
            ASSD += asd
            HD95 += hd95
    return t_acc / len(testloader),JC/len(testloader),ASSD/len(testloader),HD95/len(testloader)






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


def train_model(epoch, trainloader, optimizer_stu, device, net_stu, ema_model=None, acc=None, supervision_type='labeled', loss=None, learning_rate=None, iter_num=0, global_model=None, mu=1e-2):
    net_stu.train()
    ema_model.train()
    t_loss, t_acc = 0, 0
    max_iterations = 30000

    for i, batch in tqdm(enumerate(trainloader), total=len(trainloader), desc="Training Batches"):
        images = batch['img']
        true_masks = batch['mask']
        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.int8)
        mask_pred = net_stu(images).squeeze(1)
        masks_pred = sigmoid(mask_pred)
        masks_pred = masks_pred.float()

        if supervision_type == 'labeled':
            # Compute labeled loss (cross-entropy + dice loss)
            loss_ce = CE_Loss(masks_pred, true_masks.float())
            loss_dice = (1 - dice_coeff(masks_pred, true_masks.type(torch.float)))[0]
            loss_total = loss_dice + loss_ce

        else:
            weakEnhanceImage = torch.flip(images, dims=(2, 3))  # Flip images horizontally

            pred1 = net_stu(weakEnhanceImage).squeeze()

            diceloss = (1 - dice_coeff(sigmoid(pred1), masks_pred.squeeze().type(torch.float)))[0]
            ce_loss = CE_Loss(sigmoid(pred1), (masks_pred.squeeze().detach() > 0.5).float())
            loss_total = 0.25 * ce_loss + 0.75 * diceloss

            # Update EMA model
            update_ema_variables(net_stu, ema_model, 0.99, iter_num)

        # FedProx regularization term
        if global_model is not None:
            proximal_term = 0.0
            for w, w_t in zip(net_stu.parameters(), global_model.parameters()):
                proximal_term += ((w - w_t.detach()) ** 2).sum()
            loss_total += (mu / 2) * proximal_term

        # Learning rate decay
        lr_ = learning_rate * (1.0 - iter_num / max_iterations) ** 0.9
        for param_group in optimizer_stu.param_groups:
            param_group['lr'] = lr_

        iter_num += 1

        optimizer_stu.zero_grad()
        loss_total.backward()
        optimizer_stu.step()

        t_loss += loss_total.item()
        masks_pred = (masks_pred.detach() > 0.5).float()
        t_acc_network = dice_coeff(masks_pred, true_masks.type(torch.float)).item()
        t_acc += t_acc_network

    return t_acc / len(trainloader), t_loss / len(trainloader)


########################################
def train_fedmix(trainloader, net_stu, optimizer_stu, \
                 device, ema_net=None, supervision_type='labeled', \
                 FedMix_network=0):
    net_stu.train()

    for i, batch in enumerate(trainloader):
        imgs, masks, y_pl = batch['img'], batch['mask'], batch['y_pl']
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer_stu.zero_grad()
        ###################################################
        l_ = 0
        ## get the prediction from the model of interest ##
        masks_stu = torch.sigmoid(net_stu(imgs)).squeeze()
        ### if supervision type is labeled, just train as normal with dice ###
        if supervision_type == 'labeled':

            l_ce = CE_Loss(masks_stu.squeeze(), masks.type(torch.float).squeeze())
            l_dice = (1 - dice_coeff(masks_stu.squeeze(), masks.type(torch.float).squeeze()))[0]
            l_ = 0.25*l_stu+0.75*l_dice
        else:
            ema_net.train()
            ema_masks_stu = torch.sigmoid(ema_net(imgs)).squeeze()
            l_stu = CE_Loss(ema_masks_stu, (masks_stu.detach() > 0.5).float())
            # l_stu = (1 - dice_coeff(ema_masks_stu, (masks_stu.detach() > 0.5).float()))[0]
            l_ = l_stu
        #############################
        l_.backward()
        optimizer_stu.step()

        # for evaluation
        t_loss += l_.item()
        masks_stu = (masks_stu.detach() > 0.5).float()
        t_acc_network = dice_coeff(masks_stu, masks.type(torch.float).squeeze()).item()
        t_acc += t_acc_network


    return t_acc / len(trainloader),t_loss / len(trainloader)



def save_model(PTH, epoch, nets, acc_train):
    p_global = PTH + 'avgglobal2'
    os.makedirs(p_global, exist_ok=True)
    torch.save(nets['global'], p_global + '/tvtmodel_' + str(epoch) + '.pth')




def save_model_4(PTH,epoch, dice,nets):

    torch.save(nets['global'], os.path.join(PTH , 'fedavg_net_{}_{}.pth'.format(epoch,dice)))
    # torch.save(nets2, os.path.join(PTH , 'eam_ne_{}t.pth'.format(dice)))





def sort_rows(matrix, num_rows):
    matrix_T = torch.transpose(matrix, 0, 1)
    sorted_T = torch.topk(matrix_T, num_rows)[0]
    return torch.transpose(sorted_T, 1, 0)

