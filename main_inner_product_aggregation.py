#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid,cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, CNNFemnist, CharLSTM, CifarCnn
from models.Fed import FedWeightAvg, FedInnerAgg
from models.test import test_img
from utils.dataset import FEMNIST, ShakeSpeare


if __name__ == '__main__':

    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 加载数据集并划分客户端
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        #trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trans_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trans_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fashion-mnist':
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        dataset_test  = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                              transform=trans_fashion_mnist)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'femnist':
        dataset_train = FEMNIST(train=True)
        dataset_test = FEMNIST(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: femnist dataset is naturally non-iid')
        else:
            print("Warning: The femnist dataset is naturally non-iid, you do not need to specify iid or non-iid")
    elif args.dataset == 'shakespeare':
        dataset_train = ShakeSpeare(train=True)
        dataset_test = ShakeSpeare(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: ShakeSpeare dataset is naturally non-iid')
        else:
            print("Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid")
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # 建立模型
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.dataset == 'femnist' and args.model == 'cnn':
        net_glob = CNNFemnist(args=args).to(args.device)
    elif args.dataset == 'shakespeare' and args.model == 'lstm':
        net_glob = CharLSTM().to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')


    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    acc_test = []
    iterations = []
    accuracies = []
    clients = [LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])  # 客户端本地训练
               for idx in range(args.num_users)]
    m, clients_index_array = max(int(args.frac * args.num_users), 1), range(args.num_users)
    for iter in range(args.epochs):
        w_locals, loss_locals, weight_locols = [], [], []
        grad_locals = None
        glob_grad = None
        local_w = None
        idxs_users = np.random.choice(clients_index_array, m, replace=False)
        for idx in idxs_users:
            w, loss, grad = clients[idx].train_inneragg(net=copy.deepcopy(net_glob).to(args.device))

            # w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            # grad_locals.append(copy.deepcopy(grad))
            # weight_locols.append(len(dict_users[idx]))
            if grad_locals is None:
                grad_locals = {}
                local_w = {}
            # 将多个张量合并成一个张量
            local_grad_tensor = torch.cat([grad.flatten() for grad in grad])
            grad_locals[idx] = copy.deepcopy(local_grad_tensor)
            local_w[idx] = copy.deepcopy(w)
            # 计算全局梯度（所有客户端的本地梯度累加求均值）(f = 1/k * sum(Fk))
            if glob_grad is None:
                glob_grad = local_grad_tensor
            else:
                # 累加全局梯度
                glob_grad = glob_grad + local_grad_tensor
        glob_grad = torch.div(glob_grad, m)
        # update global weights
        # 两版聚合：全局参数+内积系数*delta_w 、 内积系数*本地参数
        w_agg = FedInnerAgg(idxs_users, local_w, grad_locals, glob_grad)
        # w_glob = w_agg
        for key in w_glob:
            w_glob[key] = w_glob[key] + w_agg[key]
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print accuracy
        net_glob.eval()
        acc_t, loss_t = test_img(net_glob, dataset_test, args)
        print("Round {:3d},Testing accuracy: {:.2f}".format(iter, acc_t))

        acc_test.append(acc_t.item())
        # 记录迭代轮数和准确率
        iterations.append(iter + 1)
        accuracies.append(acc_t)

    filename = './log/inner_agg_{}_{}_{}_C{}_acc.txt'.format(args.dataset, args.model, args.epochs, args.frac)

    with open(filename, 'w') as f:
        for i, acc in zip(iterations, accuracies):
            f.write(f"{i} {acc}\n")

    # rootpath = './log'
    # if not os.path.exists(rootpath):
    #     os.makedirs(rootpath)
    # accfile = open(rootpath + '/accfile_fed_{}_{}_{}_iid{}.dat'.
    #                format(args.dataset, args.model, args.epochs, args.iid), "w")
    #
    # for ac in acc_test:
    #     sac = str(ac)
    #     accfile.write(sac)
    #     accfile.write('\n')
    # accfile.close()
    #
    # # plot loss curve
    # plt.figure()
    # plt.plot(range(len(acc_test)), acc_test)
    # plt.ylabel('test accuracy')
    # plt.savefig(rootpath + '/fed_{}_{}_{}_C{}_iid{}_acc.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))



