from obspy.clients.filesystem.sds import Client
from obspy.core import *
from obspy.signal.trigger import *

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
        # print(net, sta, loc, chan)
        st.traces[0].stats.distance = counter
        stream += st
    except IndexError:
        pass



stream.detrend()
# stream.normalize()
# stream.plot(type="section", time_down=True)


for trace in stream:
    df = trace.stats.sampling_rate
    cft = classic_sta_lta(trace.data, int(5 * df), int(10 * df))
    plot_trigger(trace, cft, 1.5, 0.5)
