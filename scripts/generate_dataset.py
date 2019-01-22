from obspyNN.io import get_dir_list, write_training_pkl
from obspyNN.io import read_hyp_inventory, write_channel_coordinates

sds_root = "/mnt/DATA"
# sfile_list = ["/mnt/tf_data/sfile/201718select.out"]
# pkl_dir = "/mnt/tf_data/pkl/201718select"

sfile_dir = "/mnt/Data/2017_2018_sfile"
sfile_list = get_dir_list(sfile_dir, limit=100)

pkl_dir = "/mnt/tf_data/pkl/small_set"
write_training_pkl(sfile_list, sds_root, pkl_dir=pkl_dir, remove_dir=True)

hyp = "/mnt/tf_data/geom/STATION0.HYP"
inventory = read_hyp_inventory(hyp, "HL")
print(inventory)

kml_output_dir = "/mnt/tf_data/kml"
pkl_list = get_dir_list(pkl_dir)
write_channel_coordinates(pkl_list, pkl_dir + "_geom", inventory, kml_output_dir, remove_pkl_dir=True)
