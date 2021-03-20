#!/usr/bin/env python
# -*- coding:utf-8 -*-  
__author__ = 'IT小叮当'
__time__ = '2021-03-14 16:11'

import pandas as pd
import matplotlib.pyplot as plt
import time

#支持中文显示
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

df1 = pd.read_excel("G:\\AdaSGD-D\\resnet18_cmopare_cifar10\\resnet18_cifar10_adam.xlsx")
df2 = pd.read_excel("G:\\AdaSGD-D\\resnet18_cmopare_cifar10\\resnet18_cifar10_sgd.xlsx")
df3 = pd.read_excel("G:\\AdaSGD-D\\resnet18_cmopare_cifar10\\resnet18_cifar10_pid.xlsx")

def plt_curve(objective):

    plt.plot(df1[objective],label='Adam',linewidth=1.5,c='b',ls='-')
    plt.plot(df2[objective],label='SGD-M',linewidth=1.5,c='r',ls='-.')
    plt.plot(df3[objective],label='AdaSGD-D',linewidth=1.5,c='g',ls='--')

    # plt.plot(df["blue1"],df["blue2"],label='较小学习率',linewidth=3,color='b',ls='-.')
    # plt.plot(df["red1"],df["red2"],label='适当学习率',linewidth=3,color='r')
    # plt.plot(df["black1"],df["black2"],label='较大学习率',linewidth=3,color='k',ls=':')
    #plt.xlabel("Epoch")
    plt.xlabel("轮数")
    if objective == "Train Loss":
       yname = "训练损失"
       plt.ylabel(yname)
    if objective == "Valid Loss":
       yname = "验证损失"
       plt.ylabel(yname)
    if objective == "Train Acc":
       yname = "训练准确度"
       plt.ylabel(yname + "（%）")
    if objective == "Valid Acc":
       yname = "验证准确度"
       plt.ylabel(yname + "（%）")
    #plt.ylabel(objective)
    #plt.xlim(0, 20)
    plt.xticks(range(0,101,5))
    plt.legend()
    # plt.grid()
    #plt.show(bbox_inches='tight')
    plt.savefig('G:\\AdaSGD-D\\compare_cifar10\\resnet18\\'+ objective+'.jpg',dpi=300,bbox_inches='tight')
    clf()
    time.sleep(3)
    print('cifar10'+objective+'绘制完成！')

plt_curve('Train Loss')

plt_curve('Valid Loss')

plt_curve('Train Acc')

plt_curve('Valid Acc')

