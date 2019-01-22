from obspy import UTCDateTime

from obspyNN.io import get_dir_list, write_station_pkl
from obspyNN.io import read_hyp_inventory, write_channel_coordinates

sds_root = "/mnt/DATA"
pkl_output_dir = "/mnt/tf_data/pkl/scan"
xml = "/mnt/tf_data/kml/HL.xml"

start_time = UTCDateTime("2018-02-09 01:43:54")
end_time = start_time + 30
nslc = ("HL", "*", "*", "*Z")

write_station_pkl(pkl_output_dir, sds_root, nslc, start_time, end_time, remove_dir=True)

hyp = "/mnt/tf_data/geom/STATION0.HYP"
inventory = read_hyp_inventory(hyp, "HL")
print(inventory)

pkl_list = get_dir_list(pkl_output_dir)
write_channel_coordinates(pkl_list, pkl_output_dir + "_geom", inventory)
