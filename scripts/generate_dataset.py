import os
import obspyNN

sds_root = "/mnt/DATA"
sfile_dir = "/mnt/Data/2017_2018_sfile"
pkl_dir = "/mnt/tf_data/pkl/201718select"

sfile_list = ["/mnt/tf_data/sfile/201718select.out"]

if not sfile_list:
    sfile_list = []
    for file in obspyNN.io.files(sfile_dir):
        sfile_list.append(os.path.join(sfile_dir, file))
        if len(sfile_list) >= 100:
            break


os.makedirs(pkl_dir, exist_ok=True)
obspyNN.io.generate_trainning_pkl(sfile_list, sds_root, pkl_dir=pkl_dir, plot=False)


