# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from utils import *
from torch.autograd import Variable
import sys
import math
from layers import *

class Model(nn.Module):  
    def __init__(self, args, data): 
        super().__init__() 
        self.m = data.m
        self.w = args.window
        self.h = args.horizon
        self.k = args.k
        self.hw = args.hw
        self.hidA = args.hidA
        self.hidR = args.hidR
        self.hidP = args.hidP
        self.num_layers = args.n_layer
        self.dp = args.dropout
        self.dropout = nn.Dropout(p=self.dp)
        self.activate = nn.LeakyReLU()
        self.highway = nn.Linear(self.hw, 1)
        self.output = nn.Linear(self.hidA+self.hidR, 1)
        self.regionconvhid = self.k*4*self.hidP + self.k

        self.lstm = nn.LSTM(1, self.hidR, bidirectional=False, batch_first=True, num_layers=self.num_layers)
        self.rac = RegionAwareConv(P=self.w, m=self.m, k=self.k, hidP=self.hidP)
        self.q_linear = nn.Linear(self.regionconvhid, self.hidA, bias=True)
        self.k_linear = nn.Linear(self.regionconvhid, self.hidA, bias=True)
        self.v_linear = nn.Linear(self.regionconvhid, self.hidA, bias=True)
        self.attn_layer = DotAtt()
        self.inter = nn.Parameter(torch.FloatTensor(self.m, self.hidA), requires_grad=True)
        self.intra = nn.Parameter(torch.FloatTensor(self.m, self.hidR), requires_grad=True)
        nn.init.orthogonal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        self.init_weights()
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data) # best
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, x, feat=None):

        # Inter-Region embedding
        rac = self.rac(x)
        q = self.q_linear(rac)
        k = self.k_linear(rac)
        v = self.v_linear(rac)
        q = nn.Dropout(p=0.2)(q)
        k = nn.Dropout(p=0.2)(k)
        v = nn.Dropout(p=0.2)(v)
        i = self.attn_layer(q, k, v)

        # Intra-Region embedding
        r = x.permute(0, 2, 1).contiguous().view(-1, x.size(1), 1) 
        r_out, hc = self.lstm(r, None)
        last_hid = self.dropout(r_out[:,-1,:])
        t = last_hid.view(-1,self.m, self.hidR)

        # parametric-matrix fusion
        i = torch.mul(self.inter, i)
        t = torch.mul(self.intra, t)

        res = torch.cat([t, i], dim=2)
        res = self.output(res)
        res = res.squeeze(2)

        # highway
        if self.hw > 0:
            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.m)
            res = res + z
        
        return res, None