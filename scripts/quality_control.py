import numpy as np
from obspy import read

from obspyNN.io import get_dir_list
from obspyNN.plot import plot_trace

pkl_dir = "/mnt/tf_data/pkl/scan_predict"
pkl_list = get_dir_list(pkl_dir)

index = np.arange(len(pkl_list))
np.random.shuffle(index)

for i in index[0:20]:
    trace = read(pkl_list[i]).traces[0]
    plot_trace(trace)
    plot_trace(trace, enlarge=True)
