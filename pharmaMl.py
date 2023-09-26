import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

df1 = pd.read_csv("internProj\salesdaily.csv")
df2 = pd.read_csv("internProj\saleshourly.csv")
df3 = pd.read_csv("internProj\salesmonthly.csv")
df4 = pd.read_csv("internProj\salesweekly.csv")

df1['datum'] = pd.to_datetime(df1['datum'])

fig =plt.figure(figsize=(15,7))

fig.suptitle("Pharma Sales Comparision(Quantity)", fontsize =20)

ax1 = fig.add_subplot(231)
ax1.set_title('M01AB')

ax1.plot(df1["datum"],df1["M01AB"], color = 'green')

ax2 = fig.add_subplot(232)
ax2.set_title('M01AE')

ax2.plot(df1["datum"],df1["M01AE"],color = 'purple')

ax3 = fig.add_subplot(233)
ax3.set_title('N02BA')

ax3.plot(df1['datum'],df1['N02BA'], color = 'Red')

ax4 = fig.add_subplot(234)
ax4.set_title('N02BAE')

ax4.plot(df1['datum'],df1['N02BE'], color = 'Yellow')

ax5 = fig.add_subplot(235)
ax5.set_title('N05B')

ax5.plot(df1['datum'],df1['N05B'], color = 'teal')

ax6 = fig.add_subplot(236)
ax6.set_title('N05C')

ax6.plot(df1['datum'],df1['N05C'], color = 'cyan')

plt.show()