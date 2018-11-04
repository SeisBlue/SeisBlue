import matplotlib
import matplotlib.pyplot as plt
from obspy.clients.filesystem.sds import Client
from obspy.core import *
from obspy.signal.trigger import *

from mpl_toolkits import mplot3d
import pandas as pd

from sklearn import cluster

sdsRoot = "/mnt/DATA/DATA"
client = Client(sds_root=sdsRoot)
client.nslc = client.get_all_nslc(sds_type="D")
t = UTCDateTime("201602060356")
stream = Stream()
counter = 0
for net, sta, loc, chan in client.nslc:
    counter += 1
    st = client.get_waveforms(net, sta, loc, chan, t, t + 60)
    try:
        print(net, sta, loc, chan)
        st.traces[0].stats.distance = counter
        stream += st
    except IndexError:
        pass


# stream.normalize()
# stream.detrend()


def normalized(x):
    x = (x - x.mean()) / (x.max() - x.min())
    return x


# for trace in stream:
#     df = trace.stats.sampling_rate
#     cft = classic_sta_lta(trace.data, int(5 * df), int(10 * df))
#     plot_trigger(trace, cft, 1.5, 0.5)
for trace in stream:
    samp_rate = trace.stats.sampling_rate
    N = int(samp_rate)

    df = pd.DataFrame(trace.data)
    # df = pd.DataFrame(np.random.randn(35000))

    CMA = df.rolling(N, center=True).mean()
    CMA = normalized(CMA)

    POW = df.rolling(N, center=True).apply(lambda x: (x ** 2).mean())
    POW = normalized(POW)

    nsta = int(0.1 * samp_rate)
    nlta = int(1 * samp_rate)
    STALTA = classic_sta_lta_py(df, nsta, nlta)
    STALTA = pd.DataFrame(STALTA)
    STALTA = normalized(STALTA)

    idx = range(len(df[nlta:]))

    X = pd.concat([CMA[nlta:], POW[nlta:], STALTA[nlta:]], axis=1)
    X = X.dropna()

    # km = cluster.KMeans(n_clusters=5).fit(X)
    # km_cluster_labels = km.labels_
    # cluster_labels = km_cluster_labels

    db = cluster.DBSCAN(eps=0.05, min_samples=10).fit(X)
    db_cluster_labels = db.labels_
    cluster_labels = db_cluster_labels

    # sp = cluster.SpectralClustering(n_clusters=2, eigen_solver='amg').fit(X)
    # sp_cluster_labels = sp.labels_
    # cluster_labels = sp_cluster_labels

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter(CMA[nlta:], POW[nlta:], STALTA[nlta:], c=idx, cmap='rainbow', edgecolors='none', marker='.')
    ax.set_xlabel('CMA')
    ax.set_ylabel('POW')
    ax.set_zlabel('STA/LTA')
    fig.tight_layout()
    # ax.set_zlim(-1, 1)
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter(CMA[nlta:], POW[nlta:], STALTA[nlta:], c=cluster_labels, cmap='brg', edgecolors='none', marker='.')
    ax.set_xlabel('CMA')
    ax.set_ylabel('POW')
    ax.set_zlabel('STA/LTA')
    fig.tight_layout()
    # ax.set_zlim(-1, 1)
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    plt.show()

    plt.subplot(211)
    plt.scatter(idx, df[nlta:], c=idx, cmap='rainbow', edgecolors='none', marker='.')
    plt.xlabel('Index')
    plt.ylabel('Count')
    plt.xlim(0, len(df[nlta:]))

    plt.subplot(212)
    plt.plot(idx[-len(cluster_labels):], cluster_labels)
    plt.xlabel('Index')
    plt.ylabel('Count')
    plt.xlim(0, len(df[nlta:]))
    fig.tight_layout()
    plt.show()
