import matplotlib.pyplot as plt
from obspy import read
from obspy.signal.trigger import recursive_sta_lta, trigger_onset

from seisnn.io import get_dir_list

predict_pkl_dir = "/mnt/tf_data/dataset/2017_02"
predict_pkl_list = get_dir_list(predict_pkl_dir)
on = 3.5
off = 0.5
for i, pkl in enumerate(predict_pkl_list):
    trace = read(pkl).traces[0]
    start_time = trace.stats.starttime
    df = trace.stats.sampling_rate
    cft = recursive_sta_lta(trace.data, int(0.2 * df), int(2. * df))
    on_of = trigger_onset(cft, on, off)

    # Plotting the results

    ax = plt.subplot(211)
    plt.plot(trace.data, 'k')
    ymin, ymax = ax.get_ylim()
    try:
        plt.vlines(on_of[:, 0], ymin, ymax, color='r', linewidth=2)
        plt.vlines(on_of[:, 1], ymin, ymax, color='b', linewidth=2)
    except TypeError:
        pass
    plt.subplot(212, sharex=ax)
    plt.plot(cft, 'k')
    plt.hlines([on, off], 0, len(cft), color=['r', 'b'], linestyle='--')
    plt.xticks(range(0, 3001, 500), range(0, 31, 5))
    plt.xlim()
    plt.show()
