#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
import copy

def quantize_matrix(matrix, bit_width=16, alpha=0.5):
    og_sign = torch.sign(matrix)
    uns_matrix = matrix * og_sign
    uns_result = torch.round((uns_matrix * (pow(2, bit_width - 1) - 1.0) / alpha))
    result = (og_sign * uns_result)
    return result


def unquantize_matrix(matrix, bit_width=16, alpha=0.5):
    matrix = matrix.int()
    og_sign = torch.sign(matrix)
    uns_matrix = matrix * og_sign
    uns_result = uns_matrix * alpha / (pow(2, bit_width - 1) - 1.0)
    result = og_sign * uns_result
    return result.float()


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x

class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.bit_width = args.bit_width
        self.alpha = args.alpha
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)
        self.lastc1 = copy.deepcopy(self.conv1.weight.detach())
        self.lastc2 = copy.deepcopy(self.conv2.weight.detach())
        self.lastf1 = copy.deepcopy(self.fc1.weight.detach())
        self.lastf2 = copy.deepcopy(self.fc2.weight.detach())

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

    def update(self):
        self.lastc1 = copy.deepcopy(self.conv1.weight.detach())
        self.lastc2 = copy.deepcopy(self.conv2.weight.detach())
        self.lastf1 = copy.deepcopy(self.fc1.weight.detach())
        self.lastf2 = copy.deepcopy(self.fc2.weight.detach())

    def diff(self):
        temp1 = self.conv1.weight.detach() - self.lastc1
        temp2 = self.conv2.weight.detach() - self.lastc2
        temp3 = self.fc1.weight.detach() - self.lastf1
        temp4 = self.fc2.weight.detach() - self.lastf2

        self.conv1.weight = copy.deepcopy(nn.Parameter(temp1))
        self.conv2.weight = copy.deepcopy(nn.Parameter(temp2))
        self.fc1.weight = copy.deepcopy(nn.Parameter(temp3))
        self.fc2.weight = copy.deepcopy(nn.Parameter(temp4))

    def add(self):
        temp1 = self.conv1.weight + self.lastc1
        temp2 = self.conv2.weight + self.lastc2
        temp3 = self.fc1.weight + self.lastf1
        temp4 = self.fc2.weight + self.lastf2

        self.conv1.weight = copy.deepcopy(nn.Parameter(temp1))
        self.conv2.weight = copy.deepcopy(nn.Parameter(temp2))
        self.fc1.weight = copy.deepcopy(nn.Parameter(temp3))
        self.fc2.weight = copy.deepcopy(nn.Parameter(temp4))

    def quant(self, args):
        c1 = quantize_matrix(self.conv1.weight, self.bit_width, self.alpha)
        self.conv1.weight = torch.nn.Parameter(torch.FloatTensor(c1).to(args.device))

        c2 = quantize_matrix(self.conv2.weight, self.bit_width, self.alpha)
        self.conv2.weight = torch.nn.Parameter(torch.FloatTensor(c2).to(args.device))

        f1 = quantize_matrix(self.fc1.weight, self.bit_width, self.alpha)
        self.fc1.weight = torch.nn.Parameter(torch.FloatTensor(f1).to(args.device))

        f2 = quantize_matrix(self.fc2.weight, self.bit_width, self.alpha)
        self.fc2.weight = torch.nn.Parameter(torch.FloatTensor(f2).to(args.device))


    def unquant(self, args):
        c1 = unquantize_matrix(self.conv1.weight.detach(), self.bit_width, self.alpha)
        self.conv1.weight = torch.nn.Parameter(torch.FloatTensor(c1).to(args.device))

        c2 = unquantize_matrix(self.conv2.weight.detach(), self.bit_width, self.alpha)
        self.conv2.weight = torch.nn.Parameter(torch.FloatTensor(c2).to(args.device))

        f1 = unquantize_matrix(self.fc1.weight.detach(), self.bit_width, self.alpha)
        self.fc1.weight = torch.nn.Parameter(torch.FloatTensor(f1).to(args.device))

        f2 = unquantize_matrix(self.fc2.weight.detach(), self.bit_width, self.alpha)
        self.fc2.weight = torch.nn.Parameter(torch.FloatTensor(f2).to(args.device))


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNNFMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class CNNCeleba(nn.Module):
    def __init__(self, args):
        args.block = Bottleneck
        args.grayscale = False
        args.layers = [2, 2, 2, 2]
        self.inplanes = 64
        if args.grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(CNNCeleba, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(args.block, 64, args.layers[0])
        self.layer2 = self._make_layer(args.block, 128, args.layers[1], stride=2)
        self.layer3 = self._make_layer(args.block, 256, args.layers[2], stride=2)
        self.layer4 = self._make_layer(args.block, 512, args.layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1, padding=2)
        self.fc = nn.Linear(2048 * args.block.expansion, args.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return probas
