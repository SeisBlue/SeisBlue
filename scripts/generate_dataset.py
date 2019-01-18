import obspyNN

sds_root = "/mnt/DATA"
sfile_list = "/mnt/tf_data/sfile/small_list"
pkl_output = "/mnt/tf_data/pkl/small_set.pkl"

stream = obspyNN.io.get_picked_stream(sfile_list, sds_root, plot=False)
stream.write(pkl_output, format="PICKLE")
