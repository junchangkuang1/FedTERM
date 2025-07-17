# %%
import os

from torchvision import transforms
from os import listdir
from os.path import splitext
from pathlib import Path
import torch.optim as optim
import math
from unet import UNet
from IPython.display import clear_output
import re
############################
# Helper func
############################
from Fixmatch_helper import *
import numpy as np
#################################
TRAIN_RATIO = 0.9
RS = 1012  # random state
N_CHANNELS, N_CLASSES = 1, 1
bilinear = True
BATCH_SIZE, EPOCHS = 16, 200
# BATCH_SIZE, EPOCHS = 16, 200
img_size = 224
CROP_SIZE = (224, 224)
#########################################
# data_path = r'G:\thirdWork_datasets'
data_path = r'/root/autodl-tmp/project/kjc/data'
CLIENTS = ['BIDMC','I2CVB','RUNMC','UCL','HK','BMC']
# CLIENTS = ['XY','SX','GD','JM']
TOTAL_CLIENTS = len(CLIENTS)

device = torch.device('cuda:0')
LR, WD, TH = 1e-5, 1e-5, 0.9

lung_dataset = dict()
for client in CLIENTS:
    if client != 'Interobs' and client != 'Lung1':
        lung_dataset[client + '_train'] = BasicDataset(data_path, split=client, train='train',
                                                       transforms=transforms.Compose(
                                                           [RandomGenerator(output_size=CROP_SIZE, train=True)]))


        lung_dataset[client + '_test'] = BasicDataset(data_path, split=client, train='test',
                                                      transforms=transforms.Compose(
                                                          [RandomGenerator(output_size=CROP_SIZE, train=False)]))
    else:
        lung_dataset[client] = BasicDataset(data_path, split=client, train=False, transforms=transforms.Compose(
            [RandomGenerator(output_size=CROP_SIZE, train=False)]))

TOTAL_DATA = []
for client in CLIENTS:
    if client != 'Interobs' and client != 'Lung1':
        print(len(lung_dataset[client + '_train']))
        TOTAL_DATA.append(len(lung_dataset[client + '_train']))

DATA_AMOUNT = sum(TOTAL_DATA)
# WEIGHTS = [t / DATA_AMOUNT for t in TOTAL_DATA]
WEIGHTS = [0.1666,0.1666,0.1666,0.1666,0.1666,0.1666]
ORI_WEIGHTS = copy.deepcopy(WEIGHTS)

score_val = [0, 0, 0, 0,0,0]
score_val_1 = [0,0,0,0,0,0]
score_test = [0, 0, 0, 0,0,0]

training_clients, testing_clients,  valling_clients= dict(), dict(), dict()

acc_train, acc_valid, acc_test, loss_train, loss_test = dict(), dict(), dict(), dict(), dict()
loss_t = dict()
confidence = dict()
alpha_acc = []

nets, previous_nets, optimizers = dict(), dict(), dict()

acc_train = []
loss_train = []
# %%

nets['global'] = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=True).to(device)
ema_net = nets['global']


for client in CLIENTS:
    print(client)
    if client != 'Interobs' and client != 'Lung1':
        training_clients[client] = DataLoader(lung_dataset[client + '_train'], batch_size=1, shuffle=True,
                                              num_workers=0)
     ###################################################################################
        testing_clients[client] = DataLoader(lung_dataset[client + '_test'], batch_size=1, shuffle=False, num_workers=0)

        nets[client] = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=True).to(device)
        optimizers[client] = optim.Adam(nets[client].parameters(), lr=LR, weight_decay=WD)
        # optimizers[client] = optim.Adam(nets[client].parameters(), lr=LR, weight_decay=WD)
    else:
        testing_clients[client] = DataLoader(lung_dataset[client], batch_size=1, shuffle=False, num_workers=0)


for client in CLIENTS:
    if client == 'Lung1' or client == 'Interobs':
        print(client)
        print(len(lung_dataset[client]))

WEIGHTS_POSTWARMUP = [0.1666,0.1666,0.1666,0.1666,0.1666,0.1666] # put more weight to client with strong supervision
# WARMUP_EPOCH = 150
WARMUP_EPOCH = 0
CLIENTS_SUPERVISION = ['labeled', 'labeled', 'labeled','unlabeled','unlabeled','unlabeled']

# %% md
test_dice = []
all_test = []
weight = []
### First 150 epochs warmup by training locally on labeled clients
# 初始化全局模型
global_model = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=True).to(device)

def save_model_4(PTH,epoch, dice,nets):

    torch.save(nets['global'], os.path.join(PTH , 'fedpro_net_{}_{}.pth'.format(epoch,dice)))
    # torch.save(nets2, os.path.join(PTH , 'eam_ne_{}t.pth'.format(dice)))

# 聚合客户端模型的函数（FedAvg）
def aggregate_models(global_model, client_models, client_weights):
    with torch.no_grad():
        # 将全局模型的所有参数初始化为0
        for param in global_model.parameters():
            param.data.fill_(0)

        # 使用客户端的权重对模型进行加权平均
        for model, weight in zip(client_models, client_weights):
            for param, global_param in zip(model.parameters(), global_model.parameters()):
                global_param.data.add_(param.data * weight)


index = []
iter_nums = 0

USE_UNLABELED_CLIENT = False

for epoch in range(EPOCHS):
    print('epoch {} :'.format(epoch))

    if epoch == WARMUP_EPOCH:
        WEIGHTS = WEIGHTS_POSTWARMUP
        USE_UNLABELED_CLIENT = True

    index.append(epoch)

    #################### copy fed model ###################
    copy_fed(CLIENTS, nets, fed_name='global')

    epoch_train_acc, epoch_test_acc,epoch_test_hd95,epoch_test_assd,epoch_test_jc = [],[],[],[],[]
    epoch_loss = []
    #### conduct training #####
    for client, supervision_t in zip(CLIENTS, CLIENTS_SUPERVISION):
        print(client)
        if supervision_t == 'unlabeled':
            if not USE_UNLABELED_CLIENT:
                continue
        if client != 'Interobs' and client != 'Lung1':
            acc_, loss_ = train_model(
                epoch,
                training_clients[client],
                optimizers[client],
                device,
                nets[client],
                ema_model=ema_net,
                supervision_type=supervision_t,
                learning_rate=LR,
                iter_num=iter_nums,
                global_model=global_model,  # 传递全局模型
                mu=1e-2  # 可选：你可以调整 mu 的值
            )

            epoch_loss.append(loss_)
            epoch_train_acc.append(acc_)
            # 在每一轮训练后聚合客户端模型
            aggregate_models(global_model, nets.values(), WEIGHTS)

    loss_train.append(sum(epoch_loss)/len(epoch_loss))
    acc_train.append(sum(epoch_train_acc)/len(epoch_train_acc))
    aggr_fed(CLIENTS, WEIGHTS, nets)


    for order, (client, supervision_t) in enumerate(zip(CLIENTS, CLIENTS_SUPERVISION)):
        # testloader, net, device, acc = None, loss = None
        # test(epoch,valling_clients[client], nets['global'], device,supervision_t, acc_valid[client], loss_test[client])
        acc_test,jc,assd,hd95 = test(epoch,testing_clients[client], nets['global'], device, supervision_t)
        epoch_test_jc.append(jc)
        epoch_test_assd.append(assd)
        epoch_test_hd95.append(hd95)
        epoch_test_acc.append(acc_test)
    print('test score')
    print(epoch)
    print("acc is :", epoch_test_acc)
    print("jc is :", epoch_test_jc)
    print("assd is :", epoch_test_assd)
    print("hd95 is :", epoch_test_hd95)

    print("weight is::::", WEIGHTS)
    # if  epoch>100:
    #      save_model_4(r'/root/autodl-tmp/project/kjc/tn3k',epoch,sum(epoch_test_acc)/TOTAL_CLIENTS, nets)
    # # ############################################################
    # if avg_acc > best_avg_acc:
    #     print('best dice',score_test)
    #     best_avg_acc = avg_acc
    #     best_epoch = epoch
    #     save_model_4(r'C:\Users\Admin\Desktop\our model', epoch, nets, ema_net)
    # save_mode_path = "F:\pythonProject\my\model/epoch/"
    # torch.save(nets['global'].state_dict(), save_mode_path + 'epoch_' + str(epoch) + '.pth')
    # torch.save(ema_net.state_dict(), save_mode_path + 'emaepoch_' + str(epoch) + '.pth')


# with open(r'C:\Users\Admin\Desktop\onstep\second\fedavg\loss_train.txt','a') as f:
#     f.writelines(str(loss_train))
# # with open(r'C:\Users\Admin\Desktop\onstep\second\fedavg\acc_train.txt','a') as f1:
#     f1.writelines(str(acc_train))
    # np.save(r'C:\Users\Admin\Desktop\onstep\second\fedavg\loss_train.npy', loss_train)
    # np.save(PTH + '/outcome/weight', w)
    # np.save(PTH + '/outcome/score', s)
