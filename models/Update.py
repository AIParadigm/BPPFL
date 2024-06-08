#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from utils.options import args_parser
import tenseal as ts
args = args_parser()
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.lr = args.lr
        self.lr_decay = args.lr_decay

    def context(self):
        context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
        context.global_scale = pow(2, 40)
        context.generate_galois_keys()
        return context

    def decrypt(self, enc):
        return enc.decrypt().tolist()
    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=self.args.momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.lr_decay)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                # print(list(log_probs.size()))
                # print(labels)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        self.lr = scheduler.get_last_lr()[0]
        if self.args.enc :
            w = {name: param.clone().detach() for name, param in net.named_parameters()}
            context = self.context()
            enc_w = {}
            for key, value in w.items():
                value = value.cpu()
                enc_data = ts.ckks_tensor(context, value)
                enc_w[key] = enc_data
                dec_data = self.decrypt(enc_data)
            return enc_w, sum(epoch_loss) / len(epoch_loss)
        else:
            return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_inneragg(self, net):
        w_0 = {name: param.clone().detach() for name, param in net.named_parameters()}
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=self.args.momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.lr_decay)
        local_gradients = [param.clone().detach() - param.clone().detach() for param in net.parameters()]  # 创建一个全0梯度列表
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                # print(list(log_probs.size()))
                # print(labels)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                # local_gradients = local_gradients + [param.grad.clone().detach() for param in net.parameters()]
                for idx, param in enumerate(net.parameters()):
                    local_gradients[idx] += param.grad.clone().detach()
                optimizer.step()
                scheduler.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        self.lr = scheduler.get_last_lr()[0]
        w_1 = {name: param.clone().detach() for name, param in net.named_parameters()}
        delta_w = {name: w1 - w0 for name, w0, w1 in zip(w_0.keys(), w_0.values(), w_1.values())}

        # return net.state_dict(), sum(epoch_loss) / len(epoch_loss), local_gradients
        return delta_w, sum(epoch_loss) / len(epoch_loss), local_gradients

