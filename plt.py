import pandas as pd
import matplotlib.pyplot as plt


#支持中文显示
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']





df = pd.read_excel("G:\\AdaSGD-D\\3.xlsx")
#print(df)


plt.plot(df["yellow1"],df["yellow2"],label='极大学习率',linewidth=3,c='y',ls='--')
plt.plot(df["blue1"],df["blue2"],label='较小学习率',linewidth=3,color='b',ls='-.')
plt.plot(df["red1"],df["red2"],label='适当学习率',linewidth=3,color='r')
plt.plot(df["black1"],df["black2"],label='较大学习率',linewidth=3,color='k',ls=':')
plt.xlabel("迭代次数")
plt.ylabel('损失函数值')
plt.xlim((60, 350))
plt.legend()
plt.grid()
plt.show()
