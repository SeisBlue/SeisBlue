from obspyNN.io import read_geom


inventory = read_geom("/mnt/tf_data/STATION0.HYP", "HL")
for net in inventory:
    for sta in net:
        print(sta)