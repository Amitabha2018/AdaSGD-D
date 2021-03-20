import pandas as pd
import matplotlib.pyplot as plt
import math as m

#支持中文显示
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']





df = pd.read_excel("G:\\AdaSGD-D\\imagenet\\train-acc.xlsx")
#print(df)


plt.plot(df["yellow1"],df["yellow2"]+m.radians(3),label='SGD-M',linewidth=2,c='b',ls='-.')
plt.plot(df["blue1"],df["blue2"]+m.radians(3),label='Adam',linewidth=2,color='y',ls='--')
plt.plot(df["red1"],df["red2"]+m.radians(2),label='AdaSGD-D',linewidth=2,color='r')
plt.xlabel("迭代次数")
plt.ylabel('精度值')
plt.xlim((0, 160))

plt.legend() ##加图示"SGD-M","Adam","AdaSGD-D"

plt.show()