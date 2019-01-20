from obspyNN.io import get_dir_list, write_training_pkl

sds_root = "/mnt/DATA"
# sfile_list = ["/mnt/tf_data/sfile/201718select.out"]
# pkl_dir = "/mnt/tf_data/pkl/201718select"

sfile_dir = "/mnt/Data/2017_2018_sfile"
sfile_list = get_dir_list(sfile_dir, limit=100)
pkl_dir = "/mnt/tf_data/pkl/small_set"

write_training_pkl(sfile_list, sds_root, pkl_dir=pkl_dir, remove_dir=True)
