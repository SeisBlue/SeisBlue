from obspy import UTCDateTime

from seisnn.data.io import write_station_dataset

sds_root = "/mnt/DATA"
pkl_output_dir = "/mnt/tf_data/dataset/scan"
xml = "/mnt/tf_data/kml/HL.xml"

start_time = UTCDateTime("2017-04-25 23:14:28")
end_time = start_time + 30
nslc = ("HL", "*", "*", "*Z")

write_station_dataset(pkl_output_dir, sds_root, nslc, start_time, end_time, remove_dir=True)
