from obspy import read
import numpy as np
import obspyNN

result = read("/mnt/tf_data/pkl/predict_201718select.pkl")
index = np.arange(len(result))
np.random.shuffle(index)

for i in index[0:200]:
    trace = result[i]
    obspyNN.plot.plot_trace(trace)
    obspyNN.plot.plot_trace(trace, enlarge=True)
