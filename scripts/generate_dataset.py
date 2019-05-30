from seisnn.io import read_event_list, read_pkl, write_pkl, write_training_dataset

sds_root = "/mnt/DATA"
sfile_list = ["/mnt/tf_data/catalog/201718select.out"]
output_dir = "/mnt/tf_data/dataset/201718select_random"
catalog_pkl = "/mnt/tf_data/catalog/saved_catalog.pkl"

# catalog = read_event_list(sfile_list)
# write_pkl(catalog, catalog_pkl)

# catalog = read_pkl(catalog_pkl)
# cat = catalog.filter("time >= 2018-02-15")
# write_pkl(cat, "/mnt/tf_data/catalog/saved_small_catalog.pkl")

cat = read_pkl("/mnt/tf_data/catalog/saved_small_catalog.pkl")

write_training_dataset(cat, sds_root, output_dir=output_dir, batch_size=100, remove_dir=True)
