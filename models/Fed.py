#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import torch
from torch import nn
import tenseal as ts
from utils.options import args_parser

args = args_parser()

def context():
    context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = pow(2, 40)
    context.generate_galois_keys()
    return context

def decrypt(enc):
    return enc.decrypt().tolist()
def FedWeightAvg(w):
    if args.enc:
        w_avg = None
        for key, value in w.items():
            if w_avg is None:
                # 如果累加字典为空，将当前加密字典作为初始值
                w_avg = value
            else:
                for key, value1 in w_avg.items():
                    dec_value1 = decrypt(value1)
                    # 获取dict2中对应层的参数值
                    value2 = value[key]
                    dec_value2 = decrypt(value2)
                    # 将对应层的参数值相加
                    result = value1 + value2
                    w_avg[key] = result
                    dec_value = decrypt(result)
        for key, value in w_avg.items():
            value = decrypt(value)
            w_avg[key] = value
    else:
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))

    return w_avg


def FedInnerAgg(idxs_users, w_locals, grad_locals, glob_grad):
    # 计算内积
    inner_product = None  # 存储每个客户端的内积 (<f,Fk>)
    sum_inner_product = None  # 内积和 sum((<f,Fk>))
    for idx in idxs_users:
        if sum_inner_product is None:
            inner_product = {}
            inner_product[idx] = torch.dot(grad_locals[idx], glob_grad)
            if inner_product[idx] < 0:
                sum_inner_product = copy.deepcopy(-inner_product[idx])
            else:
                sum_inner_product = copy.deepcopy(inner_product[idx])
        else:
            inner_product[idx] = torch.dot(grad_locals[idx], glob_grad)
            if inner_product[idx] < 0:
                sum_inner_product -= copy.deepcopy(inner_product[idx])
            else:
                sum_inner_product += copy.deepcopy(inner_product[idx])

    # 计算每个客户端的加权平均梯度
    w_agg = None
    for idx in idxs_users:
        if w_agg is None:
            w_agg = {}
            for key, val in w_locals[idx].items():
                w_agg[key] = (inner_product[idx] / sum_inner_product) * val
        else:
            for key, val in w_locals[idx].items():
                w_agg[key] = w_agg[key] + (inner_product[idx] / sum_inner_product) * val

    return w_agg
