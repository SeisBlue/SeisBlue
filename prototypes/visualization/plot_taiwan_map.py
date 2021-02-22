import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from seisnn.io import read_hyp, read_event_list
from seisnn.utils import get_config

W, E, S, N = 119, 123, 21.5, 25.7
stamen_terrain = cimgt.Stamen('terrain-background')
fig = plt.figure(figsize=(8, 10))
ax = fig.add_subplot(1, 1, 1, projection=stamen_terrain.crs)
ax.set_extent([W, E, S, N], crs=ccrs.Geodetic())
ax.add_image(stamen_terrain, 11)

events = read_event_list('HL201718')
HL_eq = []
for event in events:
    HL_eq.append([event.origins[0].longitude, event.origins[0].latitude])
HL_eq = np.array(HL_eq).T
ax.scatter(HL_eq[0], HL_eq[1], label='Earthquake',
           transform=ccrs.Geodetic(), color='#333333', edgecolors='k', linewidth=0.3, marker='o', s=0.5)

events = read_event_list('MN2016')
MN_eq = []
for event in events:
    MN_eq.append([event.origins[0].longitude, event.origins[0].latitude])
MN_eq = np.array(MN_eq).T
ax.scatter(MN_eq[0], MN_eq[1],
           transform=ccrs.Geodetic(), color='#333333', edgecolors='k', linewidth=0.3, marker='o', s=0.5)

geom = read_hyp('HL2017.HYP')
HL_station = []
for k, station in geom.items():
    HL_station.append([station['longitude'], station['latitude']])
HL_station = np.array(HL_station).T
ax.scatter(HL_station[0], HL_station[1], label='HL station',
           transform=ccrs.Geodetic(), color='#3F51B5', edgecolors='k', linewidth=0.1, marker='v', s=5)

geom = read_hyp('HL2018.HYP')
HL_station = []
for k, station in geom.items():
    if station['latitude'] > 23.65:
        HL_station.append([station['longitude'], station['latitude'], ])
HL_station = np.array(HL_station).T
ax.scatter(HL_station[0], HL_station[1],
           transform=ccrs.Geodetic(), color='#3F51B5', edgecolors='k', linewidth=0.1, marker='v', s=5)

geom = read_hyp('MN2016.HYP')
MN_station = []
for k, station in geom.items():
    MN_station.append([station['longitude'], station['latitude'], ])
MN_station = np.array(MN_station).T
ax.scatter(MN_station[0], MN_station[1], label='MN station',
           transform=ccrs.Geodetic(), color='#b71c1c', edgecolors='k', linewidth=0.1, marker='v', s=5)

ax.add_patch(patches.Rectangle((121.3, 23.5), 0.5, 0.75,
                               transform=ccrs.Geodetic(),
                               linewidth=2,
                               edgecolor='#3F51B5',
                               facecolor='none'))

ax.add_patch(patches.Rectangle((120.1, 22.75), 0.75, 0.5,
                               transform=ccrs.Geodetic(),
                               linewidth=2,
                               edgecolor='#b71c1c',
                               facecolor='none'))

ax.set_xticks(np.arange(np.ceil(W), np.floor(E) + 1, 1), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(np.ceil(S), np.floor(N) + 1, 1), crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=True,
               bottom=True, top=True, left=True, right=True)

ax.legend(markerscale=5)

config = get_config()
plt.savefig(os.path.join(config['GEOM_ROOT'], 'TW_map.svg'), format='svg', dpi=500)
