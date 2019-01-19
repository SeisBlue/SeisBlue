import obspyNN
import shutil
import os
from obspy import UTCDateTime

sds_root = "/mnt/DATA"
pkl_dir = "/mnt/tf_data/pkl/scan"

start_time = UTCDateTime("2018-02-09 01:43:54")
end_time = start_time + 30
nslc = ("HL", "*", "*", "??Z")

shutil.rmtree(pkl_dir, ignore_errors=True)
os.makedirs(pkl_dir, exist_ok=True)

obspyNN.io.generate_station_pkl(pkl_dir, sds_root, nslc, start_time, end_time)
