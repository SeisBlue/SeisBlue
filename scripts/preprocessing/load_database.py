from seisnn.io import read_hyp, read_event_list
from seisnn.db import Client

database = "HL2018.db"
catalog = "HL2018"
geometry = "HL2018.HYP"

db = Client(database)

geom = read_hyp(geometry)
db.add_geom(geom, network="HL2018")
db.geom_summery()

events = read_event_list(catalog)
db.add_events(events)
db.plot_map()

db.add_picks(events, tag="manual", remove_duplicates=True)
db.pick_summery()
