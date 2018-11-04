from mpl_toolkits import mplot3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from obspy.core import read
from obspy.signal.trigger import classic_sta_lta_py

from sklearn import cluster


def normalized(x):
    x = (x - x.mean()) / (x.max() - x.min())
    return x


data = 'WAV/1996-06-03-2002-18S.TEST__012'

trace = read(data)[3]
trace.normalize()
trace.plot()


samp_rate = trace.stats.sampling_rate
N = int(samp_rate)

df = pd.DataFrame(trace.data)
# df = pd.DataFrame(np.random.randn(35000))


CMA = df.rolling(N, center=True).mean()
CMA = normalized(CMA)

POW = df.rolling(N, center=True).apply(lambda x: (x ** 2).mean())
POW = normalized(POW)

STALTA = classic_sta_lta_py(df, int(5 * samp_rate), int(10 * samp_rate))
STALTA = pd.DataFrame(STALTA)
STALTA = normalized(STALTA)

idx = range(len(df))

X = pd.concat([CMA, POW, STALTA], axis=1)
X = X.dropna()
kmeans_fit = cluster.KMeans(n_clusters=3).fit(X)
cluster_labels = kmeans_fit.labels_

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(CMA, POW, STALTA, c=idx, cmap='rainbow', edgecolors='none', marker='.')
ax.set_xlabel('CMA')
ax.set_ylabel('POW')
ax.set_zlabel('STA/LTA')
fig.tight_layout()
ax.set_zlim(-1, 1)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(CMA, POW, STALTA, c=cluster_labels, cmap='rainbow', edgecolors='none', marker='.')
ax.set_xlabel('CMA')
ax.set_ylabel('POW')
ax.set_zlabel('STA/LTA')
fig.tight_layout()
ax.set_zlim(-1, 1)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()


plt.subplot(211)
plt.scatter(idx, df, c=idx, cmap='rainbow', edgecolors='none', marker='.')
plt.xlabel('Index')
plt.ylabel('Count')
plt.xlim(0, len(df))

plt.subplot(212)
plt.plot(idx[-len(cluster_labels):], cluster_labels)
plt.xlabel('Index')
plt.ylabel('Count')
plt.xlim(0, len(df))
fig.tight_layout()
plt.show()

