import torch.nn as nn
import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, proj_dim=64):
        """
        Args:
            n_channels: number of input channels.
            n_classes:  number of output classes.
            bilinear:   whether to use bilinear upsampling.
            proj_dim:   feature dimension of the projection layer.
                        Used by CPL to compute class prototypes
                        and cross-client pseudo-labels.
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.bilinear   = bilinear
        self.proj_dim   = proj_dim

        # ---- Encoder ----
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64,  128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # ---- Decoder ----
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512,  256 // factor, bilinear)
        self.up3 = Up(256,  128 // factor, bilinear)
        self.up4 = Up(128,  64,            bilinear)

        # ---- Projection layer (used by CPL for prototype features) ----
        # A lightweight 1x1 conv block that maps decoder features into a
        # compact embedding space before the final classification layer.
        self.projection = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, proj_dim, kernel_size=1)
        )

        # ---- Output classifier ----
        self.outc = OutConv(proj_dim, n_classes)

        self._initialize_weights()

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

    def _decoder_forward(self, x):
        """Run the encoder and decoder up to the layer before projection."""
        x1 = self.inc(x)        # [B, 64,  H,    W]
        x2 = self.down1(x1)     # [B, 128, H/2,  W/2]
        x3 = self.down2(x2)     # [B, 256, H/4,  W/4]
        x4 = self.down3(x3)     # [B, 512, H/8,  W/8]
        x5 = self.down4(x4)     # [B, 1024//f, H/16, W/16]
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)   # [B, 64,  H,    W]
        return x

    def extract_features(self, x):
        """Return projection-layer features for CPL prototype computation.

        Returns:
            feat: tensor of shape [B, proj_dim, H, W].
        """
        decoder_out = self._decoder_forward(x)
        feat = self.projection(decoder_out)
        return feat

    def forward(self, x):
        decoder_out = self._decoder_forward(x)
        feat        = self.projection(decoder_out)
        logits      = self.outc(feat)
        return logits
