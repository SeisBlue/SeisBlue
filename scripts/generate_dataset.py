import obspyNN

sds_root = "/mnt/DATA"
sfile_list = "/mnt/tf_data/sfile/one_list"

stream = obspyNN.io.get_picked_stream(sfile_list, sds_root, plot=False)
stream.write("/mnt/tf_data/pkl/one_set.pkl", format="PICKLE")
