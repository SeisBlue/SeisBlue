import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from obspy.core import read
from obspy.signal.trigger import classic_sta_lta_py

from sklearn import cluster


def normalized(x):
    x = (x - x.mean()) / (x.max() - x.min())
    return x


data = 'WAV/1996-06-03-1917-52S.TEST__002'

trace = read(data)[0]
trace.plot()

samp_rate = trace.stats.sampling_rate
N = int(samp_rate)

df = pd.DataFrame(trace.data)
# df = pd.DataFrame(np.random.randn(35000))

df = normalized(df)

CMA = df.rolling(N, center=True).mean()
CMA = normalized(CMA)

POW = df.rolling(N, center=True).apply(lambda x: (x ** 2).mean())
POW = normalized(POW)

STALTA = classic_sta_lta_py(df, int(0.2 * samp_rate), int(2 * samp_rate))
STALTA = pd.DataFrame(STALTA)
STALTA = normalized(STALTA)

idx = range(len(df))

X = pd.concat([CMA, POW, STALTA], axis=1)
X = X.dropna()
kmeans_fit = cluster.KMeans(n_clusters=2).fit(X)
cluster_labels = kmeans_fit.labels_

plt.show()

fig = plt.figure()
ax0 = fig.add_subplot(222, projection='3d')
ax0.scatter(CMA, POW, STALTA, c=idx, cmap='rainbow', edgecolors='none', marker='.')
ax0.legend()
ax0.set_xlabel('CMA')
ax0.set_ylabel('POW')
ax0.set_zlabel('STA/LTA')
ax0.set_zlim(-1, 1)
plt.xlim(-1, 1)
plt.ylim(-1, 1)

ax1 = fig.add_subplot(224, projection='3d')
ax1.scatter(CMA, POW, STALTA, c=cluster_labels, cmap='rainbow', edgecolors='none', marker='.')
ax1.legend()
ax1.set_xlabel('CMA')
ax1.set_ylabel('POW')
ax1.set_zlabel('STA/LTA')
ax1.set_zlim(-1, 1)
plt.xlim(-1, 1)
plt.ylim(-1, 1)

ax2 = fig.add_subplot(621)
ax2.plot(idx, df)
ax2.set_xlabel('Index')
ax2.set_ylabel('Count')
plt.xlim(0, len(df))

ax3 = fig.add_subplot(623)
ax3.plot(idx, CMA)
ax3.set_xlabel('Index')
ax3.set_ylabel('CMA')
plt.xlim(0, len(df))

ax4 = fig.add_subplot(625)
ax4.plot(idx, POW)
ax4.set_xlabel('Index')
ax4.set_ylabel('POW')
plt.xlim(0, len(df))

ax5 = fig.add_subplot(627)
ax5.plot(idx, STALTA)
ax5.set_xlabel('Index')
ax5.set_ylabel('STA/LTA')
plt.xlim(0, len(df))

ax6 = fig.add_subplot(629)
ax6.scatter(idx, df, c=idx, cmap='rainbow', edgecolors='none', marker='.')
ax6.set_xlabel('Index')
ax6.set_ylabel('Count')
plt.xlim(0, len(df))

index = range(len(cluster_labels))
ax7 = fig.add_subplot(6, 2, 11)
ax7.plot(index, cluster_labels)
ax7.set_xlabel('Index')
ax7.set_ylabel('label')
plt.xlim(0, len(df))

plt.show()
