#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from torchstat import stat
from models import Nets

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, fmnist_iid, celeba_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, CNNFMnist, CNNCeleba
from models.Fed import FedAvg
from models.test import test_img
import loading_data as dataset
from models.Nets import Bottleneck
#from service.client import *
#from service.utils.DH import *


def test():
    net_glob.eval()
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Testing accuracy: {:.2f}".format(acc_test))


if __name__ == '__main__':
    # parse args
    #print('差值量化')

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fmnist':
         trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
         dataset_train = datasets.FashionMNIST('../data/fmnist/', train=True, download=True, transform=trans_mnist)
         dataset_test = datasets.FashionMNIST('../data/fmnist/', train=False, download=True, transform=trans_mnist)
         # sample users
         if args.iid:
             dict_users = fmnist_iid(dataset_train, args.num_users)
         else:
             exit('Error: only consider IID setting in CIFAR10')
    elif args.dataset == 'celeba':
        custom_transform = transforms.Compose([transforms.CenterCrop((178, 178)),
                                               transforms.Resize((128, 128)),
                                               # transforms.Grayscale(),
                                               # transforms.Lambda(lambda x: x/255.),
                                               transforms.ToTensor()])

        dataset_train = dataset.CelebaDataset(csv_path='celeba-gender-train.csv',
                                      img_dir='data/CelebA/img_align_celeba/',
                                      transform=custom_transform)

        valid_celeba= dataset.CelebaDataset(csv_path='celeba-gender-valid.csv',
                                      img_dir='data/CelebA/img_align_celeba/',
                                      transform=custom_transform)

        dataset_test = dataset.CelebaDataset(csv_path='celeba-gender-test.csv',
                                     img_dir='data/CelebA/img_align_celeba/',
                                     transform=custom_transform)

        if args.iid:
             dict_users = celeba_iid(dataset_train, args.num_users)
        else:
             exit('Error: only consider IID setting in CIFAR10')
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # initialise aggregate service
    #p = random_prime(2048)
    #g = get_generator(p)
    #clients = []
    #for i in range(args.num_users):
    #    client = Client(i, p, g, args.bit_width)
    #    clients.append(client)
    # set bit width
    #clients[0].set_bit_width()

    # fetch the public key list and compute K
    #for i in range(args.num_users):
    #    clients[i].generate_k()

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'Fmnist':
        net_glob = CNNFMnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'celeba':
        net_glob = CNNCeleba(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    #print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        #print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        test()

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

