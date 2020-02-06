import os
import numpy as np
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from seisnn.io import read_hyp, read_event_list
from seisnn.utils import get_config

W, E, S, N = 121.3, 121.8, 23.5, 24.25
stamen_terrain = cimgt.Stamen('terrain-background')
fig = plt.figure(figsize=(8, 11))
ax = fig.add_subplot(1, 1, 1, projection=stamen_terrain.crs)
ax.set_extent([W, E, S, N], crs=ccrs.Geodetic())
ax.add_image(stamen_terrain, 11)

events = read_event_list('HL201718')
HL_eq = []
for event in events:
    HL_eq.append([event.origins[0].longitude, event.origins[0].latitude])
HL_eq = np.array(HL_eq).T
ax.scatter(HL_eq[0], HL_eq[1], label='Earthquake',
           transform=ccrs.Geodetic(), color='#555555', edgecolors='k', linewidth=0.3, marker='o', s=10)

geom = read_hyp('HL2017.HYP')
HL_station = []
for k, station in geom.items():
    HL_station.append([station['longitude'], station['latitude'], ])
HL_station = np.array(HL_station).T
ax.scatter(HL_station[0], HL_station[1], label='HL 2017 station',
           transform=ccrs.Geodetic(), color='#b71c1c', edgecolors='k', marker='v', linewidth=0.5, s=60)

geom = read_hyp('HL2018.HYP')
HL_station = []
for k, station in geom.items():
    if station['latitude'] > 23.65:
        HL_station.append([station['longitude'], station['latitude'], ])
HL_station = np.array(HL_station).T
ax.scatter(HL_station[0], HL_station[1], label='HL 2018 station',
           transform=ccrs.Geodetic(), color='#FBC02D', edgecolors='k', marker='v', linewidth=0.5, s=60)

ax.set_xticks(np.arange(121.2, 121.9, 0.25), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(23.5, 24.3, 0.25), crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=True,
               bottom=True, top=True, left=True, right=True)

ax.legend(markerscale=1)

config = get_config()
plt.savefig(os.path.join(config['GEOM_ROOT'], 'HL_map.svg'), format='svg', dpi=500)
