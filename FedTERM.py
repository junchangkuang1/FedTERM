# %%

import os
import copy
import re
import math

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import matplotlib.pyplot as plt
from IPython.display import clear_output

from helper import *
from unet import UNet
from dice_loss import dice_coeff


# =========================================================
#                   Basic hyperparameters
# =========================================================
TRAIN_RATIO = 0.8
RS = 30448  # random state
N_CHANNELS, N_CLASSES = 1, 1
bilinear = True
BATCH_SIZE, EPOCHS = 16, 250
img_size = 224
CROP_SIZE = (224, 224)

LR, WD, TH = 1e-5, 1e-5, 0.9

# Dataset path and clients
data_path = r'E:\kjc\Data\jpg'
CLIENTS = ['BIDMC', 'I2CVB', 'RUNMC', 'UCL', 'HK', 'BMC']
CLIENTS_SUPERVISION = ['labeled', 'labeled', 'labeled',
                       'unlabeled', 'unlabeled', 'unlabeled']
TOTAL_CLIENTS = len(CLIENTS)

device = torch.device('cuda:0')

# =========================================================
#               TFWU aggregation hyperparameters
# =========================================================
WARMUP_EPOCH    = 150       # Length of the warm-up stage
EPSILON         = 0.05      # epsilon in Eq. (4): fixed weight for unlabeled clients
EPSILON_VAR     = 1e-6      # Numerical stability term
MIN_HISTORY_LEN = 2         # Minimum history length required for variance


# =========================================================
#                       Dataset setup
# =========================================================
lung_dataset = dict()
for client in CLIENTS:
    lung_dataset[client + '_train'] = BasicDataset(
        data_path, split=client, train=True,
        transforms=transforms.Compose(
            [RandomGenerator(output_size=CROP_SIZE, train=True)]))
    if client != 'GX':
        lung_dataset[client + '_test'] = BasicDataset(
            data_path, split=client, train=False,
            transforms=transforms.Compose(
                [RandomGenerator(output_size=CROP_SIZE, train=False)]))

TOTAL_DATA = []
for client in CLIENTS:
    if client not in ('Interobs', 'Lung1'):
        print(len(lung_dataset[client + '_train']))
        TOTAL_DATA.append(len(lung_dataset[client + '_train']))

DATA_AMOUNT = sum(TOTAL_DATA)
WEIGHTS = [t / DATA_AMOUNT for t in TOTAL_DATA]   # Initial placeholder only
ORI_WEIGHTS = copy.deepcopy(WEIGHTS)


# =========================================================
#                  Models and optimizers
# =========================================================
nets, optimizers = dict(), dict()
training_clients, testing_clients = dict(), dict()
acc_train, acc_valid = dict(), dict()
loss_train, loss_test = dict(), dict()

nets['global'] = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES,
                      bilinear=True).to(device)
ema_net = nets['global']

for client in CLIENTS:
    training_clients[client] = DataLoader(
        lung_dataset[client + '_train'], batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0)
    if client != 'GX':
        testing_clients[client] = DataLoader(
            lung_dataset[client + '_test'], batch_size=1,
            shuffle=False, num_workers=0)

    nets[client] = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES,
                        bilinear=True).to(device)
    optimizers[client] = optim.Adam(
        nets[client].parameters(), lr=LR, weight_decay=WD)

    acc_train[client], acc_valid[client] = [], []
    loss_train[client], loss_test[client] = [], []


# =========================================================
#             Training records and best metrics
# =========================================================
score = [0] * TOTAL_CLIENTS
dice  = [0] * TOTAL_CLIENTS
alpha_acc = []
acc_train1, loss_train1 = [], []
index = []
iter_nums = 0
USE_UNLABELED_CLIENT = False

best_metrics_warmup = {
    'epoch': -1, 'acc': 0.0, 'jc': 0.0,
    'assd': float('inf'), 'hd95': float('inf')}
best_metrics_post = {
    'epoch': -1, 'acc': 0.0, 'jc': 0.0,
    'assd': float('inf'), 'hd95': float('inf')}

# Only record rounds in which the client actually participated in training,
# so that variance is not polluted by unlabeled clients during warm-up.
CLIENTS_HISTORY = {client: [] for client in CLIENTS}


# =========================================================
#                       Main training loop
# =========================================================
for epoch in range(EPOCHS):
    print('epoch {} :'.format(epoch))
    epoch_train_acc, epoch_loss = [], []
    epoch_test_acc, epoch_test_jc = [], []
    epoch_test_assd, epoch_test_hd95 = [], []
    index.append(epoch)

    # Switch to the post-warmup stage. WEIGHTS will be updated below by TFWU.
    if epoch == WARMUP_EPOCH:
        USE_UNLABELED_CLIENT = True

    # Broadcast the global model to all clients
    copy_fed(CLIENTS, nets, fed_name='global')

    # Local training on each client
    for client, supervision_t in zip(CLIENTS, CLIENTS_SUPERVISION):
        if supervision_t == 'unlabeled' and not USE_UNLABELED_CLIENT:
            acc_train[client].append(0)
            loss_train[client].append(0)
            continue

        if client in ('Interobs', 'Lung1'):
            continue

        acc_, loss_ = train_model(
            epoch, training_clients[client], optimizers[client], device,
            nets[client], nets['global'], ema_model=ema_net,
            acc=acc_train[client], supervision_type=supervision_t,
            loss=loss_train[client], learning_rate=LR, iter_num=iter_nums)
        epoch_loss.append(loss_)
        epoch_train_acc.append(acc_)

    loss_train1.append(sum(epoch_loss) / len(epoch_loss))
    acc_train1.append(sum(epoch_train_acc) / len(epoch_train_acc))

    # Global aggregation using WEIGHTS computed by TFWU in the previous round
    aggr_fed(CLIENTS, WEIGHTS, nets)

    # =====================================================
    #                      Evaluation
    # =====================================================
    avg_acc = 0.0
    score = [0] * len(CLIENTS)
    client_scores = [0] * len(CLIENTS)

    for order, (client, supervision_t) in enumerate(
            zip(CLIENTS, CLIENTS_SUPERVISION)):
        if client == 'GX':
            continue

        acc_test, jc, assd, hd95 = test(
            testing_clients[client], nets['global'], device,
            acc_valid[client], loss_test[client])

        epoch_test_acc.append(acc_test)
        epoch_test_jc.append(jc)
        epoch_test_assd.append(assd)
        epoch_test_hd95.append(hd95)
        avg_acc += acc_valid[client][-1]

        # Only log rounds where this client actually trained.
        # This keeps the variance signal meaningful for TFWU.
        if supervision_t == 'labeled' or USE_UNLABELED_CLIENT:
            CLIENTS_HISTORY[client].append(acc_test)

        if supervision_t == 'labeled':
            client_scores[order] = acc_valid[client][-1]
        dice[order] = acc_valid[client][-1]

    print('test score')
    print('acc :',  epoch_test_acc)
    print('jc  :',  epoch_test_jc)
    print('assd:',  epoch_test_assd)
    print('hd95:',  epoch_test_hd95)

    avg_acc  = float(np.mean(epoch_test_acc))
    avg_jc   = float(np.mean(epoch_test_jc))
    avg_assd = float(np.mean(epoch_test_assd))
    avg_hd95 = float(np.mean(epoch_test_hd95))

    if epoch < WARMUP_EPOCH:
        if avg_acc > best_metrics_warmup['acc']:
            best_metrics_warmup.update({
                'epoch': epoch, 'acc': avg_acc, 'jc': avg_jc,
                'assd': avg_assd, 'hd95': avg_hd95})
    else:
        if avg_acc > best_metrics_post['acc']:
            best_metrics_post.update({
                'epoch': epoch, 'acc': avg_acc, 'jc': avg_jc,
                'assd': avg_assd, 'hd95': avg_hd95})

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

    # =====================================================
    #     TFWU: Temporal Fluctuations Weight Update
    #     - Implements Eqs. (2)-(4) of the paper
    #     - Labeled clients are weighted by 1 / sigma^2
    #       (more stable -> larger weight), then normalized
    #     - Unlabeled clients receive a fixed weight epsilon
    # =====================================================
    WEIGHTS_DATA = [0.0] * len(CLIENTS)
    labeled_indices   = [i for i, s in enumerate(CLIENTS_SUPERVISION)
                         if s == 'labeled']
    unlabeled_indices = [i for i, s in enumerate(CLIENTS_SUPERVISION)
                         if s == 'unlabeled']

    # 1) Compute temporal fluctuation sigma_i^2 for each labeled client
    variances = {}
    for i in labeled_indices:
        hist = CLIENTS_HISTORY[CLIENTS[i]]
        variances[i] = np.var(hist) if len(hist) >= MIN_HISTORY_LEN else None

    # 2) Convert variances to stability scores and normalize (Eqs. (2)-(3))
    if all(v is not None for v in variances.values()):
        raw_scores  = {i: 1.0 / (variances[i] + EPSILON_VAR)
                       for i in labeled_indices}
        s_sum       = sum(raw_scores.values())
        norm_scores = {i: raw_scores[i] / s_sum for i in labeled_indices}
    else:
        # Fall back to uniform weights when history is insufficient,
        # which avoids unstable updates during the first few rounds.
        n_l = len(labeled_indices)
        norm_scores = {i: 1.0 / n_l for i in labeled_indices}

    # 3) Assign aggregation weights according to the training stage (Eq. (4))
    if epoch < WARMUP_EPOCH:
        # Warm-up stage: only labeled clients participate in aggregation
        for i in labeled_indices:
            WEIGHTS_DATA[i] = norm_scores[i]
    else:
        n_u = len(unlabeled_indices)
        labeled_total = 1.0 - EPSILON * n_u
        assert labeled_total > 0, 'EPSILON * N_U must be < 1; reduce EPSILON.'
        for i in labeled_indices:
            WEIGHTS_DATA[i] = labeled_total * norm_scores[i]
        for i in unlabeled_indices:
            WEIGHTS_DATA[i] = EPSILON

    WEIGHTS = WEIGHTS_DATA
    print(f"Epoch {epoch}: Updated client weights:",
          [f"{w:.4f}" for w in WEIGHTS])

    avg_acc = avg_acc / TOTAL_CLIENTS
