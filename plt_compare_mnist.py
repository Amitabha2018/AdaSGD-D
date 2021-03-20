#!/usr/bin/env python
# -*- coding:utf-8 -*-  
__author__ = 'IT小叮当'
__time__ = '2021-03-11 11:13'
import pandas as pd
import matplotlib.pyplot as plt
import time
from matplotlib.ticker import FuncFormatter

#支持中文显示
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

df1 = pd.read_excel("G:\\AdaSGD-D\\compare_mnist\\adam.xlsx")
df2 = pd.read_excel("G:\\AdaSGD-D\\compare_mnist\\alpha_yita_t.xlsx")
df3 = pd.read_excel("G:\\AdaSGD-D\\compare_mnist\\moment.xlsx")
df4 = pd.read_excel("G:\\AdaSGD-D\\compare_mnist\\alpha_t_yita_t.xlsx")
df5 = pd.read_excel("G:\\AdaSGD-D\\compare_mnist\\original.xlsx")

# def to_percent(temp, position):
#     return '%1.0f'%(10*temp) + '%'
#print(df)
def plt_curve(objective):

    plt.plot(df1[objective],label='Adam',linewidth=1,c='b',ls='-.')
    plt.plot(df3[objective],label='SGD-M',linewidth=1,c='r',ls='--')
    # plt.plot(df2[objective],label='AdaSGD-D_Alpha_YitaT',linewidth=1,c='g',ls='-',marker='.')
    # plt.plot(df4[objective],label='AdaSGD-D_AlphaT_YitaT',linewidth=1,c='magenta',ls='-',marker='^')
    plt.plot(df5[objective], label='AdaSGD-D', linewidth=1, c='g', ls='-')
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
    #plt.xlim(0, 20)
    plt.xticks(range(0,101,5))
    plt.legend()
    # plt.grid()
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.show()
    # plt.savefig('G:\\AdaSGD-D\\compare_mnist\\'+ objective+'.jpg',dpi=300,bbox_inches='tight')
    # clf()
    # time.sleep(3)
    # print('mnist'+objective+'绘制完成！')

plt_curve('Train Loss')

plt_curve('Valid Loss')

plt_curve('Train Acc')

plt_curve('Valid Acc')
