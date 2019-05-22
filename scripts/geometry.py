from seisnn.io import get_dir_list, read_hyp_inventory, write_channel_coordinates

pkl_dir = "/mnt/tf_data/dataset/small_set"
pkl_list = get_dir_list(pkl_dir)
pkl_output_dir = pkl_dir + "_geom"

kml_output_dir = "/mnt/tf_data/kml"

inventory = read_hyp_inventory("/mnt/tf_data/geom/STATION0.HYP", "HL")
write_channel_coordinates(pkl_list, pkl_output_dir, inventory, kml_output_dir, remove_dataset_dir=True)
