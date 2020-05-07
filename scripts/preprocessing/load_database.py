from seisnn.io import read_hyp, read_event_list
from seisnn.db import Client

database = "HL2017.db"
catalog = "HL2017"
geometry = "HL2017.HYP"

db = Client(database)

geom = read_hyp(geometry)
db.add_geom(geom, network="HL2017")
db.geom_summery()
db.plot_geom()

events = read_event_list(catalog)
db.add_picks(events, tag="manual", remove_duplicates=True)
db.pick_summery()
