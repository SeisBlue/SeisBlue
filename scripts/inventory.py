from obspyNN.io import read_hyp
from obspy import read
inventory = read_hyp("/mnt/tf_data/STATION0.HYP", "HL")

#result = read("/mnt/tf_data/result.pkl")

#coord = inventory.get_coordinates(result[0].id, result[0].stats.starttime)
#print(coord)

import obspy.io.nordic.core as nordic
sfile_event = nordic.read_nordic("/mnt/tf_data/eev.out")
print()