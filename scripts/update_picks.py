import os

from obspy import read

from seisnn.io import get_dir_list, read_pkl
from seisnn.pick import get_exist_picks

pkl_dir = "/mnt/tf_data/dataset/2018_02_18_predict"
pkl_list = get_dir_list(pkl_dir)
pkl_output_dir = pkl_dir + "_updated"

pick_pkl = "/mnt/tf_data/catalog/new_pick.dataset"
pick_list = read_pkl(pick_pkl)

os.makedirs(pkl_output_dir, exist_ok=True)

for i, pkl in enumerate(pkl_list):
    trace = read(pkl).traces[0]
    new_picks = get_exist_picks(trace, pick_list)
    trace.picks.extend(new_picks)

    time_stamp = trace.stats.starttime.isoformat()
    trace.write(pkl_output_dir + '/' + time_stamp + trace.get_id() + ".dataset", format="PICKLE")

    if i % 1000 == 0:
        print("Output file... %d out of %d" % (i, len(pkl_list)))
