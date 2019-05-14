from seisnn.io import read_event_list, read_pickle, write_pickle, write_training_pkl

sds_root = "/mnt/DATA"
sfile_list = ["/mnt/tf_data/catalog/201718select.out"]
pkl_dir = "/mnt/tf_data/pkl/201718select_random"
catalog_pkl = "/mnt/tf_data/catalog/saved_catalog.pkl"

# catalog = read_event_list(sfile_list)
# write_pickle(catalog, catalog_pkl)

catalog = read_pickle(catalog_pkl)
cat = catalog.filter("time >= 2018-02-15")

# write_training_pkl(cat, sds_root, pkl_dir=pkl_dir, random_time=60, remove_dir=True)
