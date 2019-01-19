import os
import numpy as np
from obspy import read
import obspyNN

pkl_dir = "/mnt/tf_data/pkl/201718select"
pkl_list = []
for file in obspyNN.io.files(pkl_dir):
    pkl_list.append(os.path.join(pkl_dir, file))

index = np.arange(len(pkl_list))
np.random.shuffle(index)

for i in index[0:200]:
    trace = read(pkl_list[i]).traces[0]
    obspyNN.plot.plot_trace(trace)
    obspyNN.plot.plot_trace(trace, enlarge=True)
