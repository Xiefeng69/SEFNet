# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from utils import *
from torch.autograd import Variable
import sys
import math

class DotAtt(nn.Module):
    def __init__(self, attn_dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = self.dropout(attn)
        attn = self.softmax(attn)
        output = torch.bmm(attn, v)
        return output

class ConvBranch(nn.Module):
    def __init__(self,
                 m,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation_factor,
                 hidP=1,
                 isPool=True):
        super().__init__()
        self.m = m
        self.isPool = isPool
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size,1), dilation=(dilation_factor,1), bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        if self.isPool:
            self.pooling = nn.AdaptiveMaxPool2d((hidP, m))
        #self.activate = nn.Tanh()
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.conv(x))
        x = self.batchnorm(x)
        if self.isPool:
            x = self.pooling(x)
        x = x.view(batch_size, -1, self.m)
        return x

class RegionAwareConv(nn.Module):
    def __init__(self, P, m, k, hidP, dilation_factor=2):
        super(RegionAwareConv, self).__init__()
        self.P = P
        self.m = m
        self.k = k
        self.hidP = hidP
        self.conv_l1 = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=3, dilation_factor=1, hidP=self.hidP)
        self.conv_l2 = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=5, dilation_factor=1, hidP=self.hidP)
        self.conv_p1 = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=3, dilation_factor=dilation_factor, hidP=self.hidP)
        self.conv_p2 = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=5, dilation_factor=dilation_factor, hidP=self.hidP)
        self.conv_g = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=self.P, dilation_factor=1, hidP=None, isPool=False)
        self.activate = nn.Tanh()
    
    def forward(self, x):
        x = x.view(-1, 1, self.P, self.m)
        batch_size = x.shape[0]
        # local pattern
        x_l1 = self.conv_l1(x)
        x_l2 = self.conv_l2(x)
        x_local = torch.cat([x_l1, x_l2], dim=1)
        # periodic pattern
        x_p1 = self.conv_p1(x)
        x_p2 = self.conv_p2(x)
        x_period = torch.cat([x_p1, x_p2], dim=1)
        # global
        x_global = self.conv_g(x)
        # concat and activate
        x = torch.cat([x_local, x_period, x_global], dim=1).permute(0, 2, 1)
        x = self.activate(x)
        return x
