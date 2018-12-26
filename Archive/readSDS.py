import matplotlib.pyplot as plt
from obspy.clients.filesystem.sds import Client
from obspy.core import *
from obspy.signal.trigger import *

import pandas as pd


sdsRoot = "/mnt/Data/"
client = Client(sds_root=sdsRoot)
client.nslc = client.get_all_nslc(sds_type="D")
t = UTCDateTime("201801010000")
stream = Stream()

for net, sta, loc, chan in client.nslc:
    percent = client.get_availability_percentage(net, sta, loc, chan, t, t + 31536000)
    print(net, sta, loc, chan, percent)

counter = 0
for net, sta, loc, chan in client.nslc:
    counter += 1
    st = client.get_waveforms(net, sta, loc, chan, t, t + 10)
    try:
        print(net, sta, loc, chan)
        st.traces[0].stats.distance = counter
        stream += st
    except IndexError:
        pass


for trace in stream:
    samp_rate = trace.stats.sampling_rate
    N = int(samp_rate)

    df = pd.DataFrame(trace.data)

    idx = range(len(df))
    title = trace.meta.network + " " + trace.meta.station
    plt.title(title)
    plt.plot(idx, df)
    plt.xlabel('Index')
    plt.ylabel('Count')
    plt.xlim(0, len(df))
    plt.show()