import os
import numpy as np
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from seisnn.data.io import read_hyp, read_event_list
from seisnn.utils import get_config

W, E, S, N = 120.1, 120.85, 22.75, 23.25
stamen_terrain = cimgt.Stamen('terrain-background')
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, projection=stamen_terrain.crs)
ax.set_extent([W, E, S, N], crs=ccrs.Geodetic())
ax.add_image(stamen_terrain, 11)

events = read_event_list('MN2016')
MN_eq = []
for event in events:
    MN_eq.append([event.origins[0].longitude, event.origins[0].latitude])
MN_eq = np.array(MN_eq).T
ax.scatter(MN_eq[0], MN_eq[1], label='Earthquake',
           transform=ccrs.Geodetic(), color='#555555', edgecolors='k', linewidth=0.3, marker='o', s=10)

geom = read_hyp('MN2016.HYP')
MN_station = []
for k, station in geom.items():
    MN_station.append([station['longitude'], station['latitude'], ])
MN_station = np.array(MN_station).T
ax.scatter(MN_station[0], MN_station[1], label='MN station',
           transform=ccrs.Geodetic(), color='#b71c1c', edgecolors='k', marker='v', linewidth=0.5, s=60)

ax.set_xticks(np.arange(120.25, 120.8, 0.25), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(22.75, 23.3, 0.25), crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=True,
               bottom=True, top=True, left=True, right=True)

ax.legend(markerscale=1)

config = get_config()
plt.savefig(os.path.join(config['GEOM_ROOT'], 'MN_map.svg'), format='svg', dpi=500)
