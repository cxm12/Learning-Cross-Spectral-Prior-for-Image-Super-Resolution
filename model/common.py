import math

import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride=stride,
        padding=(kernel_size // 2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=True, act=nn.ReLU(True)):
        
        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        
        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
    
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        
        return res


# 2 feature 互相增强 HABrige<MM21: BridgeNet-Depth SR and Monocular Depth Estimation>
class NIRRGBBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True):
        
        super(NIRRGBBlock, self).__init__()
        self.conv1 = nn.Sequential(conv(n_feats, n_feats, kernel_size, bias=bias))
        self.conv2 = nn.Sequential(conv(n_feats*2, n_feats, kernel_size, bias=bias))
        self.conv3 = nn.Sequential(conv(n_feats, n_feats, kernel_size, bias=bias))
        
        self.pooldeconv = nn.Sequential(
            conv(n_feats, n_feats, kernel_size, stride=2, bias=bias),
            nn.ConvTranspose2d(n_feats, n_feats, 2, 2, padding=0)
            # output = (input-1)stride+outputpadding -2padding+kernelsize
        )
        self.prelu = nn.Sequential(nn.PReLU())
        self.CA = CA(conv, n_feats, kernel_size, bias)
        self.SA = SA(conv, n_feats, kernel_size, bias)
    
    def forward(self, rgb, nir):
        nir1 = self.conv1(nir)
        pd = self.pooldeconv(nir1)
        
        pr = self.prelu(pd-nir1)
        pr1 = pr * nir1 + nir1
        
        cat = torch.cat([rgb, pr1], 1)
        cat1 = self.conv2(cat)
        catca, _ = self.CA(cat1)
        cat2 = self.conv3(catca)
        catsa, _ = self.SA(cat2)

        return catsa


class NIRRGBBlock1(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True):
        super(NIRRGBBlock1, self).__init__()
        self.n_feats = n_feats
        self.conv1 = nn.Sequential(conv(n_feats, n_feats, kernel_size, bias=bias))
        
        self.CA = CA(conv, n_feats * 2, kernel_size, bias)
        self.SA = SA(conv, n_feats, kernel_size, bias)
    
    def forward(self, rgb, nir):
        rgbnir = torch.cat([rgb, nir], 1)
        rgbnir1, cnr = self.CA(rgbnir)
        [cr, cn] = torch.split(cnr, [self.n_feats, self.n_feats], dim=1)
        
        rgb1, Srgb = self.SA(rgb)
        # nir1 = nir * Srgb
        fuse = rgb1 * cr + nir * Srgb * cn
        
        return fuse


class CA(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True):
        super(CA, self).__init__()
        
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)))
        self.conv1 = nn.Sequential(conv(n_feats, n_feats, kernel_size, bias=bias))
        self.sig = nn.Sequential(nn.Sigmoid())

    def forward(self, x):
        x1 = self.pool(x)
        x2 = self.conv1(x1)
        w = self.sig(x2)
        out = x * w
        
        return out, w


class SA(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True):
        super(SA, self).__init__()
        
        self.conv1 = nn.Sequential(conv(n_feats, 1, kernel_size, bias=bias))
        self.sig = nn.Sequential(nn.Sigmoid())
    
    def forward(self, x):
        x1 = self.conv1(x)
        w = self.sig(x1)
        out = x * w
        
        return out, w


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
        
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError
        
        super(Upsampler, self).__init__(*m)
