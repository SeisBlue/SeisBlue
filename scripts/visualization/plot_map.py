import numpy as np
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from seisnn.io import read_geom, read_event_list


W, E, S, N = 119, 123, 21.5, 25.7
stamen_terrain = cimgt.Stamen('terrain-background')
fig = plt.figure(figsize=(8, 10))
ax = fig.add_subplot(1, 1, 1, projection=stamen_terrain.crs)
ax.set_extent([W, E, S, N], crs=ccrs.Geodetic())
ax.add_image(stamen_terrain, 9)


events = read_event_list('HL201718')
for event in events:
    ax.scatter(event.origins[0].longitude, event.origins[0].latitude, transform=ccrs.Geodetic(), color='#000000', marker='o',s=0.1)

events = read_event_list('MN2016')
for event in events:
    ax.scatter(event.origins[0].longitude, event.origins[0].latitude, transform=ccrs.Geodetic(), color='#000000', marker='o',s=0.1)

geom = read_geom('HL201718.HYP')
for k, station in geom.items():
    ax.scatter(station['longitude'], station['latitude'], transform=ccrs.Geodetic(), color='#3F51B5', marker='v',s=5)

geom = read_geom('MN2016.HYP')
for k, station in geom.items():
    ax.scatter(station['longitude'], station['latitude'], transform=ccrs.Geodetic(), color='#b71c1c', marker='v',s=5)


ax.set_xticks(np.arange(np.ceil(W), np.floor(E)+1, 1), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(np.ceil(S), np.floor(N)+1, 1), crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
plt.show()


