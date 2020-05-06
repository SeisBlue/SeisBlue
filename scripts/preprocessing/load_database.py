from seisnn.io import read_hyp, read_event_list
from seisnn.sql import Client, Picks

from itertools import groupby
from operator import attrgetter

database = "test.db"
catalog = "test"
geometry = "HL2017.HYP"
tag = "manual"

geom = read_hyp(geometry)
events = read_event_list(catalog)
db = Client(database)

db.add_geom(geom)
query = db.get_geom()
for i in query:
    print(i)

db.add_picks(events, tag)
query = db.get_picks().order_by(Picks.station)
listings = [list(g) for k, g in groupby(query, attrgetter('station'))]
for i in listings:
    print(i[0].station, len(i))

query = db.remove_duplicate_picks()

count = 0
for i in query:
    count += 1
    print(i)
print(count)
