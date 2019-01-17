from obspy import read
import numpy as np
import obspyNN

result = read("/mnt/tf_data/pkl/one_set.pkl")
index = np.arange(len(result))
np.random.shuffle(index)

for i in index[0:20]:
    trace = result[i]
    obspyNN.plot.plot_trace(trace)
    obspyNN.plot.plot_trace(trace, enlarge=True)
