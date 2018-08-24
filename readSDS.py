from obspy.clients.filesystem.sds import Client
from obspy.core import *
import numpy as np
import matplotlib.pyplot as plt

sdsRoot = "/media/512ssd/home/jimmy/DATA"
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
        print(net, sta, loc, chan, "no data")


stream.detrend()
stream.normalize()
stream.plot(type="section", time_down=True)
