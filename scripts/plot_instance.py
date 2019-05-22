import numpy as np
from obspy import read

from seisnn.io import get_dir_list
from seisnn.plot import plot_dataset

# pkl_dir = "/mnt/tf_data/dataset/2018_02_18_random_predict"
pkl_dir = "/mnt/tf_data/dataset/2018_02_18_predict"
# pkl_dir = "/mnt/tf_data/dataset/scan_predict"

pkl_list = get_dir_list(pkl_dir)

index = np.arange(len(pkl_list))
# np.random.shuffle(index)

plot_dir = "/mnt/tf_data/plot"
for i in index[:200]:
    trace = read(pkl_list[i]).traces[0]
    plot_dataset(trace)
    # plot_trace(trace, save_dir=plot_dir)
    # plot_trace(trace, enlarge=True, save_dir=plot_dir + "/enlarge")


