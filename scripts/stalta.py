import matplotlib.pyplot as plt
from obspy import read
from obspy.signal.trigger import recursive_sta_lta, trigger_onset

from seisnn.io import get_dir_list

predict_pkl_dir = "/mnt/tf_data/pkl/2017_02"
predict_pkl_list = get_dir_list(predict_pkl_dir)
for i, pkl in enumerate(predict_pkl_list):
    trace = read(pkl).traces[0]
    df = trace.stats.sampling_rate
    cft = recursive_sta_lta(trace.data, int(0.5 * df), int(1. * df))
    on_of = trigger_onset(cft, 1, 0.5)

    # Plotting the results
    ax = plt.subplot(211)
    plt.plot(trace.data, 'k')
    ymin, ymax = ax.get_ylim()
    plt.vlines(on_of[:, 0], ymin, ymax, color='r', linewidth=2)
    plt.vlines(on_of[:, 1], ymin, ymax, color='b', linewidth=2)
    plt.subplot(212, sharex=ax)
    plt.plot(cft, 'k')
    plt.hlines([3.5, 0.5], 0, len(cft), color=['r', 'b'], linestyle='--')
    plt.axis('tight')
    plt.show()
