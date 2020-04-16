import argparse

from seisnn.io import read_hyp, read_event_list
from seisnn.pick import get_pick_dict
from seisnn.database import Client

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--database', required=True, help='sql database file', type=str)
ap.add_argument('-c', '--catalog', required=True, help='catalog s-file dir', type=str)
ap.add_argument('-g', '--geometry', required=True, help='geometry STATION0.HYP', type=str)
ap.add_argument('-p', '--name', required=True, help='output pickset name', type=str)
args = ap.parse_args()

geom = read_hyp(args.geometry)
events = read_event_list(args.catalog)
pick_dict = get_pick_dict(events)

db = Client(args.database)
db.add_geom(geom)
db.add_picks(pick_dict, args.name)
