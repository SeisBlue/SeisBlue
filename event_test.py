from obspy import *

cat = read_events("REA/EVENT/1996/06/03-1955-35D.S199606")
cat.plot()