from obspy import read

from seisnn.data.io import write_pkl
from seisnn.utils import get_dir_list
from seisnn.plot import plot_dataset

pick_pkl = "/mnt/tf_data/catalog/new_pick.dataset"
predict_pkl_dir = "/mnt/tf_data/dataset/2018_02_18_predict"
predict_pkl_list = get_dir_list(predict_pkl_dir)

new_pick = []
try:
    for i, pkl in enumerate(predict_pkl_list):
        trace = read(pkl).traces[0]
        start_time = trace.stats.starttime
        for p in trace.picks:
            if p.evaluation_mode == "automatic":
                if not p.evaluation_status == "confirmed" and not p.evaluation_status == "rejected":
                    pick_time = p.time - start_time
                    plot_dataset(trace, xlim=(pick_time - 2, pick_time + 4))
                    key_in = input("Confirmed pick? [y/N]")
                    if key_in == "y" or key_in == "Y":
                        print("Current file... %d out of %d" % (i, len(predict_pkl_list)))
                        p.evaluation_mode = "manual"
                        p.evaluation_status = "reviewed"
                        new_pick.append(p)

        if i % 100 == 0:
            print("Current file... %d out of %d" % (i, len(predict_pkl_list)))

except KeyboardInterrupt:
    print("Keyboard interrupt.")
finally:
    write_pkl(new_pick, pick_pkl)
    print("Pick saved.")
