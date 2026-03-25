# %%

import os
import copy

import numpy as np
import torch
import torch.nn as nn
import math
from torchvision import transforms
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from helper import *
from unet import UNet
from dice_loss import dice_coeff

import matplotlib.pyplot as plt
from IPython.display import clear_output
import re
############################
# Helper func
############################

#################################
TRAIN_RATIO = 0.8
RS = 30448  # random state
N_CHANNELS, N_CLASSES = 1, 1
bilinear = True
BATCH_SIZE, EPOCHS = 16, 100
# BATCH_SIZE, EPOCHS = 16, 100
img_size = 224
CROP_SIZE = (224, 224)
#########################################
data_path = r'C:\Users\kjc\Desktop\jpg'
# data_path = r'G:\thirdWork_datasets\second'
# data_path = r'/root/autodl-tmp'
# CLIENTS = ['XY','SX','GD','JM']
CLIENTS = ['BIDMC','I2CVB','RUNMC','UCL','HK','BMC']
TOTAL_CLIENTS = len(CLIENTS)

device = torch.device('cuda:0')
LR, WD, TH = 1e-5, 1e-5, 0.9

lung_dataset = dict()
for client in CLIENTS:
    lung_dataset[client + '_train'] = BasicDataset(data_path, split=client, train=True,
                                                   transforms=transforms.Compose(
                                                       [RandomGenerator(output_size=CROP_SIZE, train=True)]))
    if client != 'GX':
        lung_dataset[client + '_test'] = BasicDataset(data_path, split=client, train=False,
                                                  transforms=transforms.Compose(
                                                          [RandomGenerator(output_size=CROP_SIZE, train=False)]))


# %% md

## Initialize the weights

# %%

TOTAL_DATA = []
for client in CLIENTS:
    if client != 'Interobs' and client != 'Lung1':
        print(len(lung_dataset[client + '_train']))
        TOTAL_DATA.append(len(lung_dataset[client + '_train']))

DATA_AMOUNT = sum(TOTAL_DATA)
WEIGHTS = [t / DATA_AMOUNT for t in TOTAL_DATA]

ORI_WEIGHTS = copy.deepcopy(WEIGHTS)

score = [0, 0, 0, 0,0,0]
dice = [0, 0, 0, 0,0,0]
# %% md

# storage file

# %%

training_clients, testing_clients = dict(), dict()

acc_train, acc_valid, loss_train, loss_test = dict(), dict(), \
                                              dict(), dict()
loss_test = dict()
alpha_acc = []

nets, optimizers = dict(), dict()

acc_train1 = []
loss_train1 = []
# %%

nets['global'] = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=True).to(device)
ema_net = nets['global']
# for param in ema_net.parameters():
#     param.requires_grad = True
#     param.detach_()

for client in CLIENTS:

    training_clients[client] = DataLoader(lung_dataset[client + '_train'], batch_size=16, shuffle=True,
                                          num_workers=0)

    if client != 'GX':
        testing_clients[client] = DataLoader(lung_dataset[client + '_test'], batch_size=1, shuffle=False, num_workers=0)

    nets[client] = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=True).to(device)

    optimizers[client] = optim.Adam(nets[client].parameters(), lr=LR, weight_decay=WD)

    acc_train[client], acc_valid[client] = [], []
    loss_train[client], loss_test[client] = [], []

for client in CLIENTS:
    if client == 'Lung1' or client == 'Interobs':
        print(client)
        print(len(lung_dataset[client]))


# WEIGHTS_POSTWARMUP = [0.3,0.3,0.325,0.025,0.025,0.025]  # put more weight to client with strong supervision
WEIGHTS_POSTWARMUP=2
WARMUP_EPOCH =2
CLIENTS_SUPERVISION = ['labeled', 'labeled','labeled','unlabeled','unlabeled','unlabeled']
# CLIENTS_SUPERVISION = ['labeled', 'labeled']
# %% md

### First 150 epochs warmup by training locally on labeled clients

# %%

best_avg_acc, best_epoch_avg = 0, 0
index = []
iter_nums = 0

USE_UNLABELED_CLIENT = False
loss = []

best_metrics_warmup = {
    'epoch': -1,
    'acc': 0.0,
    'jc': 0.0,
    'assd': float('inf'),
    'hd95': float('inf')
}
best_metrics_post = {
    'epoch': -1,
    'acc': 0.0,
    'jc': 0.0,
    'assd': float('inf'),
    'hd95': float('inf')
}
for epoch in range(EPOCHS):

    epoch_train_acc, epoch_test_acc,epoch_test_hd95,epoch_test_assd,epoch_test_jc = [],[],[],[],[]
    epoch_loss = []

    print('epoch {} :'.format(epoch))
    if epoch == WARMUP_EPOCH:
        WEIGHTS = WEIGHTS_POSTWARMUP
        USE_UNLABELED_CLIENT = True

    index.append(epoch)

    #################### copy fed model ###################
    copy_fed(CLIENTS, nets, fed_name='global')

    #### conduct training #####
    for client, supervision_t in zip(CLIENTS, CLIENTS_SUPERVISION):
        if supervision_t == 'unlabeled':
            if not USE_UNLABELED_CLIENT:
                acc_train[client].append(0)
                loss_train[client].append(0)
                continue

        if client != 'Interobs' and client != 'Lung1':
            acc_, loss_ = train_model(epoch, training_clients[client], optimizers[client], device, nets[client],
                                      nets['global'], ema_model=ema_net, acc=acc_train[client], loss=loss_train[client],
                                      supervision_type=supervision_t, learning_rate=LR, iter_num=iter_nums)
            epoch_loss.append(loss_)
            epoch_train_acc.append(acc_)
    loss_train1.append(sum(epoch_loss)/len(epoch_loss))
    acc_train1.append(sum(epoch_train_acc)/len(epoch_train_acc))

    aggr_fed(CLIENTS, WEIGHTS, nets)
    ################### test ################################
    avg_acc = 0.0
    CLIENTS_HISTORY = {client: [] for client in CLIENTS}  # 确保 CLIENTS_HISTORY 存在

    for order, (client, supervision_t) in enumerate(zip(CLIENTS, CLIENTS_SUPERVISION)):
        # testloader, net, device, acc = None, loss = None
        if client == 'GX':
            continue
        # 如果没有 ema_model

        acc_test, jc, assd, hd95 = test(testing_clients[client], nets['global'], device, acc_valid[client],
                                        loss_test[client])

        epoch_test_jc.append(jc)
        epoch_test_assd.append(assd)
        epoch_test_hd95.append(hd95)
        epoch_test_acc.append(acc_test)
        avg_acc += acc_valid[client][-1]
        CLIENTS_HISTORY[client].append(acc_test)  # 记录历史准确率
        # if not USE_UNLABELED_CLIENT:
        if supervision_t == "labeled":
            score[order] = acc_valid[client][-1]
        # else:
        dice[order] = acc_valid[client][-1]
    ######################################################
    ####### dynamic weighting #########
    ###################################
    print('test score')
    print("acc is :", epoch_test_acc)
    print("jc is :", epoch_test_jc)
    print("assd is :", epoch_test_assd)
    print("hd95 is :", epoch_test_hd95)
    avg_acc = np.mean(epoch_test_acc)
    avg_jc = np.mean(epoch_test_jc)
    avg_assd = np.mean(epoch_test_assd)
    avg_hd95 = np.mean(epoch_test_hd95)
    if epoch < WARMUP_EPOCH:
        if avg_acc > best_metrics_warmup['acc']:
            best_metrics_warmup.update({
                'epoch': epoch,
                'acc': avg_acc,
                'jc': avg_jc,
                'assd': avg_assd,
                'hd95': avg_hd95
            })
    else:
        if avg_acc > best_metrics_post['acc']:
            best_metrics_post.update({
                'epoch': epoch,
                'acc': avg_acc,
                'jc': avg_jc,
                'assd': avg_assd,
                'hd95': avg_hd95
            })

    # 打印最佳结果
    print(f"\n[Warmup Best @ Epoch {best_metrics_warmup['epoch']}]: "
          f"Acc: {best_metrics_warmup['acc']:.4f}, "
          f"JC: {best_metrics_warmup['jc']:.4f}, "
          f"ASSD: {best_metrics_warmup['assd']:.4f}, "
          f"HD95: {best_metrics_warmup['hd95']:.4f}")

    print(f"[Post-Warmup Best @ Epoch {best_metrics_post['epoch']}]: "
          f"Acc: {best_metrics_post['acc']:.4f}, "
          f"JC: {best_metrics_post['jc']:.4f}, "
          f"ASSD: {best_metrics_post['assd']:.4f}, "
          f"HD95: {best_metrics_post['hd95']:.4f}\n")
    epsilon = 1e-6  # 避免除零

    # 初始化权重列表
    WEIGHTS_DATA = [0.0] * len(CLIENTS)

    # 计算每个客户端的方差
    variances = [np.var(history) for history in CLIENTS_HISTORY.values()]

    if epoch < 2:
        # 前100轮：无标签客户端权重固定为0
        for i, client in enumerate(CLIENTS):
            if CLIENTS_SUPERVISION[i] == "labeled":
                # 有标签客户端权重 = 方差倒数
                WEIGHTS_DATA[i] = 1 / (variances[i] + epsilon)
        # 归一化权重
        total = sum(WEIGHTS_DATA)
        WEIGHTS = [w / total for w in WEIGHTS_DATA]

    else:
        # 第100轮以后
        # 无标签客户端固定权重 0.05
        for i, client in enumerate(CLIENTS):
            if CLIENTS_SUPERVISION[i] == "unlabeled":
                WEIGHTS_DATA[i] = 0.05

        # 有标签客户端权重占剩余部分 = 1 - 无标签总权重
        remaining_weight = 1.0 - sum(WEIGHTS_DATA)
        labeled_indices = [i for i, s in enumerate(CLIENTS_SUPERVISION) if s == "labeled"]
        # 方差倒数得分
        scores = [1 / (variances[i] + epsilon) for i in labeled_indices]
        # 归一化得分
        scores_sum = sum(scores)
        scores = [s / scores_sum for s in scores]
        # 分配剩余权重
        for idx, score in zip(labeled_indices, scores):
            WEIGHTS_DATA[idx] = score * remaining_weight

        WEIGHTS = WEIGHTS_DATA

    print(f"Epoch {epoch}: Updated client weights:", WEIGHTS)

    w = []
    s = []
    w.append(WEIGHTS)
    s.append(score)

    avg_acc = avg_acc / TOTAL_CLIENTS
    # save_model_4(r'C:\Users\Admin\Desktop\ourmodel',avg_acc, epoch, nets, ema_net)
    # if epoch == 0 or epoch >200:
         # save_model_4(r'C:\Users\Admin\Desktop\onstep\second\FedDUS',sum(epoch_test_acc)/TOTAL_CLIENTS, epoch, nets)
        # save_model_4(r'/root/autodl-tmp/FedDUS',sum(epoch_test_acc)/TOTAL_CLIENTS, epoch, nets)
    ############################################################
    # if avg_acc > best_avg_acc:
    #     best_avg_acc = avg_acc
    #     best_epoch = epoch
    #     save_model_4(PTH, epoch, nets, ema_net)
    # save_mode_path = "F:\pythonProject\FedDUS\model/epoch/"
    # torch.save(nets['global'].state_dict(), save_mode_path + 'epoch_' + str(epoch) + '.pth')
    # torch.save(ema_net.state_dict(), save_mode_path + 'emaepoch_' + str(epoch) + '.pth')
# with open(r'/root/autodl-tmp/FedDUS/loss_train.txt', 'a') as f:
#     f.writelines(str(loss_train1))
# with open(r'/root/autodl-tmp/FedDUS/onstep\acc_train.txt', 'a') as f1:
#     f1.writelines(str(acc_train1))
# with open(r'C:\Users\Admin\Desktop\onstep\second/FedDUS/dus_loss_train.txt', 'a') as f:
#     f.writelines(str(loss_train1))
# with open(r'C:\Users\Admin\Desktop\onstep\second\FedDUS/dus_acc_train.txt', 'a') as f1:
#     f1.writelines(str(acc_train1))
    # with open(r"F:\pythonProject\FedDUS\model\loss.txt",'w') as f:
    #     f.writelines(loss)
    ################################
    # plot #########################
    ################################
    # np.save(PTH + '/outcome/acc_train', acc_train)
    # np.save(PTH + '/outcome/acc_test', acc_valid)
    # np.save(PTH + '/outcome/loss_train', loss_train)
    # np.save(PTH + '/outcome/weight', w)
    # np.save(PTH + '/outcome/score', s)
    # clear_output(wait=True)
