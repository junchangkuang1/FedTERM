""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.dropout_p1 = nn.Dropout2d(p=0.3)
        self.dropout_p2 = nn.Dropout2d(p=0.2)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self._initialize_weights()

        # 用于保存特征图
        self.features = None

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x): # 16 1 224 224
        x1 = self.inc(x)   # 16 64 224 224
        x2 = self.down1(x1) # 16 128 112 112 
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        # x = self.dropout_p1(x)
        x = self.up2(x, x3)
        # x = self.dropout_p2(x)
        x = self.up3(x, x2)
        # x = self.dropout_p2(x)
        x = self.up4(x, x1)
        self.features = x
        # x = self.dropout_p2(x)
        logits = self.outc(x)

          # 选择要提取的特征图层
        return logits



    