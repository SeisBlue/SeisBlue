from obspy import UTCDateTime

from obspyNN.io import write_station_pkl

sds_root = "/mnt/DATA"
pkl_output_dir = "/mnt/tf_data/pkl/scan"

start_time = UTCDateTime("2018-02-09 01:43:54")
end_time = start_time + 30
nslc = ("HL", "*", "*", "??Z")

write_station_pkl(pkl_output_dir, sds_root, nslc, start_time, end_time, remove_dir=True)
