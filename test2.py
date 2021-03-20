#!/usr/bin/env python
# -*- coding:utf-8 -*-  
__author__ = 'IT小叮当'
__time__ = '2021-03-12 22:42'
import torch    # 导入torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #如果有GPU则调用GPU 没有则调用CPU
a = torch.Tensor([1.])    # 定义变量
b = torch.Tensor([2.])
a.to(device)
b.to(device)
print(a+b)

