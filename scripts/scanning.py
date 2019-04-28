from obspy import UTCDateTime

from seisnn.io import write_station_pkl

sds_root = "/mnt/DATA"
pkl_output_dir = "/mnt/tf_data/pkl/scan"
xml = "/mnt/tf_data/kml/HL.xml"

start_time = UTCDateTime("2018-02-14 12:00:00")
end_time = start_time + 300
nslc = ("HL", "*", "*", "*Z")

write_station_pkl(pkl_output_dir, sds_root, nslc, start_time, end_time, remove_dir=True)
