from seisnn.io import write_training_pkl

sds_root = "/mnt/DATA"
sfile_list = ["/mnt/tf_data/sfile/201718select.out"]
pkl_dir = "/mnt/tf_data/pkl/201718select"

write_training_pkl(sfile_list, sds_root, pkl_dir=pkl_dir, remove_dir=True)

