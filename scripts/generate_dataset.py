import obspyNN

sds_root = "/mnt/DATA"
sfile_list = "/mnt/tf_data/sfile/201718_list"
pkl_output = "/mnt/tf_data/pkl/201718select.pkl"

stream = obspyNN.io.get_picked_stream(sfile_list, sds_root, plot=False)
stream.write(pkl_output, format="PICKLE")
