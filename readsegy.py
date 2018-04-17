import numpy as np
import matplotlib.pyplot as plt
from obspy.io.segy.segy import _read_segy
import geom


stream = _read_segy('A:\Autopicking\MCS\seismic\OR1-446\mcs446-1111adata2.segy', headonly=True)

chan = 52
data = np.stack(t.data for t in stream.traces)

vm = np.percentile(data, 98)

plt.figure(figsize=(16,6))
plt.imshow(data.T, cmap="RdGy", vmin=-vm, vmax=vm, aspect='auto')
plt.colorbar()
plt.show()