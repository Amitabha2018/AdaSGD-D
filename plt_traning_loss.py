#!/usr/bin/env python
# -*- coding:utf-8 -*-  
__author__ = 'IT小叮当'
__time__ = '2021-03-09 21:37'
import pandas as pd
import matplotlib.pyplot as plt


#支持中文显示
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']





df = pd.read_excel("G:\\AdaSGD-D\\moment.xlsx")
#print(df)


plt.plot(df["Train_Loss"],label='SGD-M',linewidth=2,c='b',ls='-',marker='o')
# plt.plot(df["blue1"],df["blue2"],label='较小学习率',linewidth=3,color='b',ls='-.')
# plt.plot(df["red1"],df["red2"],label='适当学习率',linewidth=3,color='r')
# plt.plot(df["black1"],df["black2"],label='较大学习率',linewidth=3,color='k',ls=':')
plt.xlabel("Epochx")
# plt.ylabel('损失函数值')
#plt.xlim(0, 20)
plt.xticks(range(0,21,3))
plt.legend()
plt.grid()
plt.show()
