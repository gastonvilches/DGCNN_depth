#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        idx = knn(x, k=k)
        
    #device = torch.device("cuda")
    device = torch.device(idx.device.type)

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    
    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    
    x = x.transpose(2, 1)#.contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.reshape(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)#.contiguous()
    
  
    return feature


class DGCNN_simple(nn.Module):
    def __init__(self, k=20, emb_dims=128, dropout=0.2, output_channels=1):
        super().__init__()

        self.k = k
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.bn5 = nn.BatchNorm1d(emb_dims)
        self.bn6 = nn.BatchNorm2d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(4, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False), self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(128, emb_dims, kernel_size=1, bias=False), self.bn5, nn.LeakyReLU(negative_slope=0.2))        
        self.conv6 = nn.Sequential(nn.Conv2d(emb_dims+128, 256, kernel_size=1, bias=False), self.bn6, nn.Dropout(p=dropout), nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Sequential(nn.Conv2d(256, 1, kernel_size=1, bias=False))
        
    def forward(self, x):
        batch_size = x.size(0)
        sample_point_count = x.size(2)
        
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
               
        x_concat = torch.cat((x1, x2), dim=1)
        
        x_conv5 = self.conv5(x_concat)
        
        x1 = F.adaptive_max_pool1d(x_conv5, 1).view(batch_size, -1)
        x = x1 
        
        # Repeat x,n times. 
        x_n_repeated = x.repeat(sample_point_count,1,1)
        
        # Cat with multiscale local features 
        x__cat_multiscale = torch.cat((x_n_repeated.transpose_(0, 1).transpose_(1, 2), x_concat), 1)
        
        # MPL up to n x 1        
        x = self.conv6(x__cat_multiscale.unsqueeze(3))
        
        x = self.conv9(x)
        
        return x.view(batch_size,-1)

class DGCNN_cls(nn.Module):
    def __init__(self, k=20, emb_dims=1024, dropout=0.5, output_channels=40):
        super().__init__()
        self.k = k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(4, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        
        return x
    
class DGCNN_reg(nn.Module):
    def __init__(self, k=20, emb_dims=1024, dropout=0.5, output_channels=1):
        super().__init__()
        
        self.k = k
        
        eps = 1e-05
        self.bn1 = nn.BatchNorm2d(64, eps=eps)
        self.bn2 = nn.BatchNorm2d(64, eps=eps)
        self.bn3 = nn.BatchNorm2d(64, eps=eps)
        self.bn4 = nn.BatchNorm2d(64, eps=eps)
        self.bn5 = nn.BatchNorm2d(64, eps=eps)
        self.bn6 = nn.BatchNorm1d(emb_dims, eps=eps)
        self.bn7 = nn.BatchNorm1d(512, eps=eps)
        self.bn8 = nn.BatchNorm1d(256, eps=eps)

        self.conv1 = nn.Sequential(nn.Conv2d(4, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=dropout)
        self.conv9 = nn.Conv1d(256, output_channels, kernel_size=1, bias=False)
        

    def forward(self, x, x0=None):
        batch_size = x.size(0)
        num_points = x.size(2)
        
        if x0 == None:
            x0 = get_graph_feature(x, k=self.k)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x0)                       # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        
        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
    
        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        
        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        
        x = x.repeat(1, 1, num_points)          # (batch_size, emb_dims, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, emb_dims+64*3, num_points)

        x = self.conv7(x)                       # (batch_size, emb_dims+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)

        x = x.view(batch_size,-1)
        
        x = torch.sigmoid(x)
        # a = 0.01
        # x = torch.min( torch.max( a*(x+2), 1/2+x/4 ), 1+a*(x-2) )

        return x