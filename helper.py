import os
import copy
import random
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.nn import BCEWithLogitsLoss, BCELoss, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomCrop
from torchvision.transforms.functional import adjust_gamma as intensity_shift

from PIL import Image
from scipy import ndimage
from scipy.ndimage import zoom
from medpy import metric

from dice_loss import dice_coeff


sigmoid = nn.Sigmoid()
CE_Loss = BCELoss()

###############################################
#  Constants
###############################################
colors = ['r', 'g', 'b', 'c', 'k', 'y', 'm', 'c']


# =========================================================
#  Federated aggregation utilities
# =========================================================
def aggr_fed(CLIENTS, WEIGHTS_CL, nets, fed_name='global'):
    """Weighted aggregation of client model parameters into the global model."""
    for param_tensor in nets[fed_name].state_dict():
        tmp = None
        for client, w in zip(CLIENTS, WEIGHTS_CL):
            if client in ('Interobs', 'Lung422'):
                continue
            if tmp is None:
                tmp = copy.deepcopy(w * nets[client].state_dict()[param_tensor])
            else:
                tmp += w * nets[client].state_dict()[param_tensor]
        nets[fed_name].state_dict()[param_tensor].data.copy_(tmp)
        del tmp


def copy_fed(CLIENTS, nets, fed_name='global'):
    """Broadcast the global model weights to every client."""
    for client in CLIENTS:
        if 'Interobs' not in client and 'Lung422' not in client:
            nets[client].load_state_dict(copy.deepcopy(nets[fed_name].state_dict()))


# =========================================================
#  Optional: a simple Transformer-style aggregator
#  (kept for compatibility; not used by FedTERM main loop)
# =========================================================
class TransformerAggregator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.fc = nn.Linear(hidden_dim, input_dim)
        self.input_dim = input_dim

    def forward(self, local_model_output, global_model_output):
        # Use the global model output as Query, local model output as Key/Value
        local_model_output  = local_model_output.view(-1, 1, self.input_dim)
        global_model_output = global_model_output.view(-1, 1, self.input_dim)
        attn_output, _ = self.attention(global_model_output,
                                        local_model_output,
                                        local_model_output)
        return self.fc(attn_output)


# =========================================================
#  Dataset and data augmentation
# =========================================================
class BasicDataset(Dataset):
    def __init__(self, base_dir: str, split, train=False, transforms=None):
        print(split)
        self.transform   = transforms
        self.split       = split
        self._base_dir   = base_dir
        self.train       = train
        self.image_list  = []

        list_file = '{}_train.txt'.format(split) if train else '{}_test.txt'.format(split)
        with open(os.path.join(self._base_dir, list_file), 'r') as f:
            self.image_list = f.readlines()
        self.image_list = [item.replace('\n', '') for item in self.image_list]

        print("{} has total {} samples".format(split, len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        img_path  = os.path.join(self._base_dir, self.split, 'images', image_name)
        mask_path = os.path.join(self._base_dir, self.split, 'masks',  image_name)

        image = np.array(Image.open(img_path).convert('L'))
        mask  = np.array(Image.open(mask_path).convert('L'))

        # Min-max normalization
        denom = np.max(image) - np.min(image)
        img   = (image - np.min(image)) / denom if denom > 0 else image.astype(np.float32)
        denom = np.max(mask) - np.min(mask)
        mask  = (mask - np.min(mask)) / denom if denom > 0 else mask.astype(np.float32)
        mask[mask > 0] = 1
        mask[mask < 0] = 0

        sample = {'img': img, 'mask': mask, 'filename': os.path.basename(img_path)}
        if self.transform:
            sample = self.transform(sample)
        return sample


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis  = np.random.randint(0, 2)
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
        self.train       = train

    def __call__(self, sample):
        img, mask, filename = sample['img'], sample['mask'], sample['filename']
        if self.train:
            if random.random() > 0.5:
                img, mask = random_rot_flip(img, mask)
            elif random.random() > 0.5:
                img, mask = random_rotate(img, mask)

        x, y = img.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            img  = zoom(img,  (self.output_size[0] / x, self.output_size[1] / y), order=0)
            mask = zoom(mask, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        mask[mask >= 1] = 1
        img  = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        mask = torch.from_numpy(mask.astype(np.float32))
        return {'img': img, 'mask': mask, 'filename': filename}


# =========================================================
#  Evaluation
# =========================================================
def test(testloader, net, device, acc=None, loss=None):
    net.eval()
    t_loss, t_acc = 0.0, 0.0
    JC, ASSD, HD95 = 0.0, 0.0, 0.0
    n = 0

    with torch.no_grad():
        for batch in testloader:
            image     = batch['img'].to(device=device, dtype=torch.float32)
            mask_true = batch['mask'].to(device=device, dtype=torch.float32)

            mask_pred      = net(image)
            mask_pred_norm = torch.sigmoid(mask_pred.squeeze(1))
            mask_pred_bin  = (mask_pred_norm > 0.5).float()

            # Fallback: empty prediction -> use the highest-confidence pixel
            if torch.sum(mask_pred_bin) == 0:
                top = torch.max(mask_pred_norm)
                mask_pred_bin = (mask_pred_norm == top).float()

            t = mask_true.squeeze().cpu().numpy()
            p = mask_pred_bin.squeeze().cpu().numpy()

            # Skip metric computation for fully-background ground truth
            if t.sum() == 0:
                continue

            t_acc += metric.binary.dc(t, p)
            JC    += metric.binary.jc(t, p)
            ASSD  += metric.binary.asd(t, p)
            HD95  += metric.binary.hd95(t, p)
            n     += 1

    n = max(n, 1)
    if acc is not None:
        acc.append(t_acc / n)
    if loss is not None:
        loss.append(t_loss / n)
    return t_acc / n, JC / n, ASSD / n, HD95 / n


# =========================================================
#  EMA update for the teacher / global momentum model
# =========================================================
def update_ema_variables(model, ema_model, alpha, global_step):
    """Update teacher (EMA) parameters from the student parameters."""
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


# =========================================================
#  Pseudo-label utilities for FedTERM
# =========================================================
def _binary_to_two_class_prob(p_fg, eps=1e-6):
    """Convert binary foreground probability [B, H, W] to a 2-class
    probability map [B, 2, H, W] (background, foreground)."""
    if p_fg.dim() == 4:
        p_fg = p_fg.squeeze(1)
    p_fg = torch.clamp(p_fg, eps, 1.0 - eps)
    return torch.stack([1.0 - p_fg, p_fg], dim=1)


def local_entropy(prob, kernel_size=4):
    """Compute the local entropy map LE(x, y) defined in Eqs. (7)-(8).

    Args:
        prob: probability map of shape [B, C, H, W]; sums to 1 over C.
        kernel_size: neighborhood size k.

    Returns:
        Local entropy map of shape [B, 1, H, W].
    """
    prob = torch.clamp(prob, 1e-6, 1.0)
    # Eq. (7): pixel-wise entropy
    entropy = -torch.sum(prob * torch.log(prob), dim=1, keepdim=True)
    # Eq. (8): k x k average pooling
    pad = kernel_size // 2
    return F.avg_pool2d(entropy, kernel_size=kernel_size, stride=1, padding=pad)


def loss_local_entropy(p1, p2, kernel_size=4):
    """Neighborhood Structure Entropy Loss (NSEL), Eq. (9).

    Both p1 and p2 must be 2-class (or C-class) probability maps,
    shape [B, C, H, W].
    """
    le1 = local_entropy(p1, kernel_size)
    le2 = local_entropy(p2, kernel_size)
    return torch.mean(torch.abs(le1 - le2))


@torch.no_grad()
def compute_class_prototypes(net, dataloader, device, n_classes=2,
                             max_batches=None):
    """Compute one feature prototype per class on a labeled client.

    Each client's UNet is expected to expose ``extract_features(x)`` that
    returns the projection-layer feature map of shape [B, C, H, W].

    Args:
        net: client model (in eval mode).
        dataloader: labeled training dataloader.
        device: torch device.
        n_classes: number of classes (2 for binary).
        max_batches: optional cap on the number of batches used.

    Returns:
        prototypes: [n_classes, C] tensor.
    """
    net.eval()
    feat_sum   = None
    feat_count = None

    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        images = batch['img'].to(device, dtype=torch.float32)
        masks  = batch['mask'].to(device, dtype=torch.long)  # [B, H, W]

        feats = net.extract_features(images)  # [B, C, H, W]
        B, C, H, W = feats.shape

        # Resize labels to feature spatial size
        if masks.shape[-2:] != (H, W):
            masks = F.interpolate(masks.unsqueeze(1).float(), size=(H, W),
                                  mode='nearest').squeeze(1).long()

        feats_flat  = feats.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        labels_flat = masks.reshape(-1)                         # [B*H*W]

        if feat_sum is None:
            feat_sum   = torch.zeros(n_classes, C, device=device)
            feat_count = torch.zeros(n_classes, device=device)

        for c in range(n_classes):
            mask_c = (labels_flat == c)
            if mask_c.any():
                feat_sum[c]   += feats_flat[mask_c].sum(dim=0)
                feat_count[c] += mask_c.sum().float()

    if feat_sum is None:
        return None
    feat_count = feat_count.clamp(min=1.0).unsqueeze(1)
    return feat_sum / feat_count   # [n_classes, C]


def aggregate_prototypes(prototype_list, weights=None):
    """Aggregate per-client class prototypes into global cluster centers.

    Args:
        prototype_list: list of [n_classes, C] tensors (one per labeled client).
        weights: optional list of scalar weights; defaults to uniform.

    Returns:
        aggregated prototypes of shape [n_classes, C].
    """
    valid = [p for p in prototype_list if p is not None]
    if len(valid) == 0:
        return None
    if weights is None:
        weights = [1.0 / len(valid)] * len(valid)
    out = torch.zeros_like(valid[0])
    for p, w in zip(valid, weights):
        out = out + w * p
    return out


def cpl_pseudo_label(features, cluster_centers, sigma=0.05):
    """Generate cross-client pseudo-labels via Eqs. (5)-(6).

    Args:
        features: feature map [B, C, H, W] from the projection layer.
        cluster_centers: [K, C] aggregated cluster centers across clients.
        sigma: temperature parameter.

    Returns:
        pseudo_prob: [B, K, H, W] probabilistic pseudo-labels.
    """
    B, C, H, W = features.shape
    K = cluster_centers.shape[0]
    feat_flat = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
    centers   = cluster_centers.unsqueeze(0).expand(B, -1, -1)        # [B, K, C]
    distances = torch.cdist(feat_flat, centers)                       # [B, H*W, K]
    pseudo    = F.softmax(-distances / sigma, dim=-1)                 # [B, H*W, K]
    return pseudo.permute(0, 2, 1).reshape(B, K, H, W)                # [B, K, H, W]


# =========================================================
#  Local training for FedTERM
# =========================================================
def train_model(epoch, trainloader, optimizer_stu, device, net_stu, net_glb,
                ema_model=None, acc=None, supervision_type='labeled',
                loss=None, learning_rate=None, iter_num=0,
                cluster_centers=None, sigma=0.05,
                kernel_size=4, lambda_entropy=0.25,
                max_iterations=30000):
    """Local training for FedTERM.

    For labeled clients:    L_sup   = 0.25 * L_CE + 0.75 * L_Dice              (Eq. 10)
    For unlabeled clients:  L_unsup = L_Dice(student, CPL) + lambda * L_NSEL    (Eq. 11)
                            L_NSEL is the NSEL between GPL (global model) and CPL.

    Args:
        cluster_centers: [K, C] aggregated cross-client cluster centers.
                         Required when ``supervision_type == 'unlabeled'``.
        sigma: temperature parameter in Eq. (6).
        kernel_size: neighborhood size k in Eq. (8).
        lambda_entropy: weight of the NSEL term in Eq. (11).
    """
    net_stu.train()
    if ema_model is not None:
        ema_model.train()

    t_loss, t_acc = 0.0, 0.0

    for i, batch in enumerate(trainloader):
        images     = batch['img'].to(device=device, dtype=torch.float32)
        true_masks = batch['mask'].to(device=device, dtype=torch.float32)

        # Student forward pass
        mask_pred  = net_stu(images).squeeze(1)
        masks_pred = torch.sigmoid(mask_pred)            # [B, H, W]

        if supervision_type == 'labeled':
            # ---- Supervised loss, Eq. (10) ----
            loss_ce   = CE_Loss(masks_pred, true_masks)
            loss_dice = (1 - dice_coeff(masks_pred, true_masks))[0]
            loss_total = 0.25 * loss_ce + 0.75 * loss_dice

        else:
            # ---- Semi-supervised loss with CPL, Eq. (11) ----
            if cluster_centers is None:
                # No cross-client centers yet (e.g., first round after warm-up).
                # Fall back to entropy minimization on the global model output.
                with torch.no_grad():
                    gpl_fg = torch.sigmoid(net_glb(images).squeeze(1))
                gpl_2c = _binary_to_two_class_prob(gpl_fg)
                stu_2c = _binary_to_two_class_prob(masks_pred)
                loss_total = loss_local_entropy(gpl_2c, stu_2c,
                                                kernel_size=kernel_size)
            else:
                # 1) Extract features from the projection layer
                features = net_stu.extract_features(images)  # [B, C, H, W]

                # 2) CPL pseudo-labels via cross-client cluster centers
                centers = cluster_centers.to(device=device, dtype=features.dtype)
                cpl_prob = cpl_pseudo_label(features, centers, sigma=sigma)
                # For binary segmentation: foreground channel index = 1
                cpl_fg = cpl_prob[:, 1, :, :]

                # 3) GPL: global model prediction
                with torch.no_grad():
                    gpl_fg = torch.sigmoid(net_glb(images).squeeze(1))

                # 4) Match spatial size if the projection layer downsamples
                if cpl_fg.shape[-2:] != masks_pred.shape[-2:]:
                    cpl_prob = F.interpolate(cpl_prob,
                                             size=masks_pred.shape[-2:],
                                             mode='bilinear',
                                             align_corners=False)
                    cpl_fg = cpl_prob[:, 1, :, :]

                # 5) L_DSC between student prediction and CPL foreground prob
                loss_dice = (1 - dice_coeff(masks_pred, cpl_fg.detach()))[0]

                # 6) NSEL between GPL and CPL (both 2-class probability maps)
                gpl_2c = _binary_to_two_class_prob(gpl_fg)
                cpl_2c = cpl_prob if cluster_centers.shape[0] == 2 \
                                  else _binary_to_two_class_prob(cpl_fg)
                loss_nsel = loss_local_entropy(gpl_2c, cpl_2c.detach(),
                                               kernel_size=kernel_size)

                loss_total = loss_dice + lambda_entropy * loss_nsel

            # EMA update for the teacher / global momentum model
            if ema_model is not None:
                update_ema_variables(net_stu, ema_model, 0.99, iter_num)

        # Poly learning-rate schedule
        if learning_rate is not None:
            lr_ = learning_rate * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer_stu.param_groups:
                param_group['lr'] = lr_

        optimizer_stu.zero_grad()
        loss_total.backward()
        optimizer_stu.step()
        iter_num += 1

        t_loss += loss_total.item()
        masks_pred_bin = (masks_pred.detach() > 0.5).float()
        t_acc += dice_coeff(masks_pred_bin, true_masks).item()

    n = max(len(trainloader), 1)
    if acc is not None:
        acc.append(t_acc / n)
    if loss is not None:
        loss.append(t_loss / n)
    return t_acc / n, t_loss / n


# =========================================================
#  Reference: FedMix-style training (kept for compatibility)
# =========================================================
def train_fedmix(trainloader, net_stu, optimizer_stu,
                 device, acc=None, loss=None,
                 supervision_type='labeled', FedMix_network=1):
    net_stu.train()
    t_loss, t_acc = 0.0, 0.0

    for i, batch in enumerate(trainloader):
        imgs, masks, y_pl = batch['img'], batch['mask'], batch['y_pl']
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer_stu.zero_grad()

        masks_stu = torch.sigmoid(net_stu(imgs))
        if supervision_type == 'labeled':
            l_ = (1 - dice_coeff(masks_stu, masks.float()))[0]
        else:
            masks_teach = (y_pl if FedMix_network == 1 else masks).to(device)
            l_ = (1 - dice_coeff(masks_stu, masks_teach.float()))[0]

        print("dice is :", l_.item())
        l_.backward()
        optimizer_stu.step()

        t_loss += l_.item()
        t_acc  += dice_coeff((masks_stu.detach() > 0.5).float(), masks.float()).item()

    n = max(len(trainloader), 1)
    if acc is not None:
        try:    acc.append(t_acc / n)
        except: acc.append(0.0)
    if loss is not None:
        try:    loss.append(t_loss / n)
        except: loss.append(0.0)


# =========================================================
#  Model checkpointing
# =========================================================
def save_model_4(PTH, dice, epoch, nets):
    torch.save(nets['global'],
               os.path.join(PTH, 'feddus_{}_net_{}.pth'.format(epoch, dice)))


def save_model_ll(PTH, epoch, nets, CLIENTS):
    for client in CLIENTS:
        p_global = os.path.join(PTH, 'llglobal', client)
        os.makedirs(p_global, exist_ok=True)
        torch.save(nets[client],
                   os.path.join(p_global, 'tvtmodel_' + str(epoch) + '.pth'))


def save_model_centralize(PTH, epoch, nets):
    p_global = os.path.join(PTH, 'cenglobal2')
    os.makedirs(p_global, exist_ok=True)
    torch.save(nets, os.path.join(p_global, 'tvtmodel_' + str(epoch) + '.pth'))


def sort_rows(matrix, num_rows):
    matrix_T = torch.transpose(matrix, 0, 1)
    sorted_T = torch.topk(matrix_T, num_rows)[0]
    return torch.transpose(sorted_T, 1, 0)
