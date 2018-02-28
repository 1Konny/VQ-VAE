''' This code is from :
        https://github.com/jzbontar/pixelcnn-pytorch
'''

import os
import time
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, cuda, backends
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets, transforms, utils
backends.cudnn.benchmark = True


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class PIXELCNN(nn.Module):
    # 1 chanel PixelCNN
    def __init__(self, k_dim, z_dim, kernel_size=3, fm=32):
        super(PIXELCNN, self).__init__()
        self.k_dim = k_dim
        self.z_dim = z_dim
        self.fm = fm
        self.kernel_size = kernel_size
        self.padding = (kernel_size-1)//2
        self.encode = nn.Sequential(
            MaskedConv2d('A', self.z_dim,  self.fm, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(self.fm), nn.ReLU(True),
            MaskedConv2d('B', self.fm, self.fm, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(self.fm), nn.ReLU(True),
            MaskedConv2d('B', self.fm, self.fm, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(self.fm), nn.ReLU(True),
            MaskedConv2d('B', self.fm, self.fm, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(self.fm), nn.ReLU(True),
            MaskedConv2d('B', self.fm, self.fm, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(self.fm), nn.ReLU(True),
            MaskedConv2d('B', self.fm, self.fm, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(self.fm), nn.ReLU(True),
            MaskedConv2d('B', self.fm, self.fm, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(self.fm), nn.ReLU(True),
            MaskedConv2d('B', self.fm, self.fm, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(self.fm), nn.ReLU(True),
            nn.Conv2d(self.fm, self.k_dim, 1))

    def forward(self, z):
        return self.encode(z)

if __name__ == '__main__':
    fm = 64
    net = nn.Sequential(
        MaskedConv2d('A', 1,  fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        nn.Conv2d(fm, 10, 1))


    bs, c, w, h = 5, 1, 8, 8
    image = Variable(torch.rand(bs,c,w,h)).mul_(2).add_(-1)
    out = net(image)
    import ipdb; ipdb.set_trace()