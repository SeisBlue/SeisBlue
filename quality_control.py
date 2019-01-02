from obspy import read
import numpy as np
from obspy.core import Stream
import obspyNN

result = read("/mnt/tf_data/result.pkl")
index = np.arange(len(result))
np.random.shuffle(index)

st = Stream()
for i in index[0:50]:
    st += result[i]

obspyNN.plot.plot_stream(st)



