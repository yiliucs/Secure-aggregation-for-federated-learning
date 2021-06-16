import torch
from torch import nn
import torch.nn.functional as F
import copy


def quantize_matrix(matrix, bit_width=8, alpha=0.1):
    og_sign = torch.sign(matrix)
    uns_matrix = matrix * og_sign
    uns_result = torch.round((uns_matrix * (pow(2, bit_width - 1) - 1.0) / alpha))
    result = (og_sign * uns_result)
    return result


def unquantize_matrix(matrix, bit_width=8, alpha=0.1):
    matrix = matrix.int()
    og_sign = torch.sign(matrix)
    uns_matrix = matrix * og_sign
    uns_result = uns_matrix * alpha / (pow(2, bit_width - 1) - 1.0)
    result = og_sign * uns_result
    return result.float()


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, args, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bit_width = args.bit_width
        self.alpha = args.alpha
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
        self.lastc1 = copy.deepcopy(self.conv1.weight.detach())
        self.lastc2 = copy.deepcopy(self.conv2.weight.detach())
        self.lastc3 = copy.deepcopy(self.conv3.weight.detach())


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

    def update(self):
        self.lastc1 = copy.deepcopy(self.conv1.weight.detach())
        self.lastc2 = copy.deepcopy(self.conv2.weight.detach())
        self.lastc3 = copy.deepcopy(self.conv3.weight.detach())


    def diff(self):
        temp1 = self.conv1.weight.detach() - self.lastc1
        temp2 = self.conv2.weight.detach() - self.lastc2
        temp3 = self.conv3.weight.detach() - self.lastc3


        self.conv1.weight = copy.deepcopy(nn.Parameter(temp1))
        self.conv2.weight = copy.deepcopy(nn.Parameter(temp2))
        self.conv3.weight = copy.deepcopy(nn.Parameter(temp3))


    def add(self):
        temp1 = self.conv1.weight + self.lastc1
        temp2 = self.conv2.weight + self.lastc2
        temp3 = self.conv3.weight + self.lastc3

        self.conv1.weight = copy.deepcopy(nn.Parameter(temp1))
        self.conv2.weight = copy.deepcopy(nn.Parameter(temp2))
        self.conv3.weight = copy.deepcopy(nn.Parameter(temp3))


    def quant(self, args):
        c1 = quantize_matrix(self.conv1.weight, self.bit_width, self.alpha).type(torch.FloatTensor)
        #self.conv1.weight = torch.nn.Parameter(c1).to(args.device)
        self.conv1.weight = torch.nn.Parameter(torch.FloatTensor(c1).to(args.device))

        c2 = quantize_matrix(self.conv2.weight, self.bit_width, self.alpha).type(torch.FloatTensor)
        #self.conv2.weight = torch.nn.Parameter(c2).to(args.device)
        self.conv2.weight = torch.nn.Parameter(torch.FloatTensor(c2).to(args.device))

        c3 = quantize_matrix(self.conv3.weight, self.bit_width, self.alpha).type(torch.FloatTensor)
        #self.conv3.weight = torch.nn.Parameter(c3).to(args.device)
        self.conv3.weight = torch.nn.Parameter(torch.FloatTensor(c3).to(args.device))


    def unquant(self, args):
        c1 = unquantize_matrix(self.conv1.weight.detach(), self.bit_width, self.alpha).type(torch.FloatTensor)
        #self.conv1.weight = torch.nn.Parameter(c1).to(args.device)
        self.conv1.weight = torch.nn.Parameter(torch.FloatTensor(c1).to(args.device))

        c2 = unquantize_matrix(self.conv2.weight.detach(), self.bit_width, self.alpha).type(torch.FloatTensor)
        #self.conv2.weight = torch.nn.Parameter(c2).to(args.device)
        self.conv2.weight = torch.nn.Parameter(torch.FloatTensor(c2).to(args.device))

        c3 = quantize_matrix(self.conv3.weight, self.bit_width, self.alpha).type(torch.FloatTensor)
        #self.conv3.weight = torch.nn.Parameter(c3).to(args.device)
        self.conv3.weight = torch.nn.Parameter(torch.FloatTensor(c3).to(args.device))




class QCNNCeleba(nn.Module):
    def __init__(self, args):
        args.block = Bottleneck
        args.grayscale = False
        args.layers = [2, 2, 2, 2]
        self.bit_width = args.bit_width
        self.alpha = args.alpha
        self.inplanes = 64
        if args.grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(QCNNCeleba, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(args.block, 64, args.layers[0], args)
        self.layer2 = self._make_layer(args.block, 128, args.layers[1], args, stride=2)
        self.layer3 = self._make_layer(args.block, 256, args.layers[2], args, stride=2)
        self.layer4 = self._make_layer(args.block, 512, args.layers[3], args, stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1, padding=2)
        self.fc = nn.Linear(2048 * args.block.expansion, args.num_classes)
        self.lastc1 = copy.deepcopy(self.conv1.weight.detach())
        self.lastf1 = copy.deepcopy(self.fc.weight.detach())



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, args, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, args, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, args))

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

    def update(self):
        self.lastc1 = copy.deepcopy(self.conv1.weight.detach())
        self.lastf1 = copy.deepcopy(self.fc.weight.detach())
        for layer in self.layer1:
            layer.update()
        for layer in self.layer2:
            layer.update()
        for layer in self.layer3:
            layer.update()
        for layer in self.layer4:
            layer.update()

    def diff(self):
        temp1 = self.conv1.weight.detach() - self.lastc1
        temp2 = self.fc.weight.detach() - self.lastf1

        self.conv1.weight = copy.deepcopy(nn.Parameter(temp1))
        self.fc.weight = copy.deepcopy(nn.Parameter(temp2))

        for layer in self.layer1:
            layer.diff()
        for layer in self.layer2:
            layer.diff()
        for layer in self.layer3:
            layer.diff()
        for layer in self.layer4:
            layer.diff()


    def add(self):
        temp1 = self.conv1.weight + self.lastc1
        temp2 = self.fc.weight + self.lastf1

        self.conv1.weight = copy.deepcopy(nn.Parameter(temp1))
        self.fc.weight = copy.deepcopy(nn.Parameter(temp2))

        for layer in self.layer1:
            layer.add()
        for layer in self.layer2:
            layer.add()
        for layer in self.layer3:
            layer.add()
        for layer in self.layer4:
            layer.add()


    def quant(self, args):
        c1 = quantize_matrix(self.conv1.weight, self.bit_width, self.alpha).type(torch.FloatTensor) 
        #self.conv1.weight = torch.nn.Parameter(c1).to(args.device)
        self.conv1.weight = torch.nn.Parameter(torch.FloatTensor(c1).to(args.device))

        f1 = quantize_matrix(self.fc.weight, self.bit_width, self.alpha).type(torch.FloatTensor) 
        #self.fc.weight = torch.nn.Parameter(f1).to(args.device)
        self.fc.weight = torch.nn.Parameter(torch.FloatTensor(f1).to(args.device))

        for layer in self.layer1:
            layer.quant(args)
        for layer in self.layer2:
            layer.quant(args)
        for layer in self.layer3:
            layer.quant(args)
        for layer in self.layer4:
            layer.quant(args)

    def unquant(self, args):
        c1 = unquantize_matrix(self.conv1.weight, self.bit_width, self.alpha).type(torch.FloatTensor) 
        #self.conv1.weight = torch.nn.Parameter(c1).to(args.device)
        self.conv1.weight = torch.nn.Parameter(torch.FloatTensor(c1).to(args.device))

        f1 = unquantize_matrix(self.fc.weight, self.bit_width, self.alpha).type(torch.FloatTensor) 
        #self.fc.weight = torch.nn.Parameter(f1).to(args.device)
        self.fc.weight = torch.nn.Parameter(torch.FloatTensor(f1).to(args.device))

        for layer in self.layer1:
            layer.unquant(args)
        for layer in self.layer2:
            layer.unquant(args)
        for layer in self.layer3:
            layer.unquant(args)
        for layer in self.layer4:
            layer.unquant(args)

