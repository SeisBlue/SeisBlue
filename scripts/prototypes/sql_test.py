import argparse

from seisnn.io import read_hyp, read_event_list
from seisnn.pick import get_pick_dict
from seisnn.database import add_geom, add_picks

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--database', required=True, help='database file', type=str)
ap.add_argument('-c', '--catalog', required=True, help='catalog s-file dir', type=str)
ap.add_argument('-g', '--geometry', required=True, help='geometry STATION0.HYP', type=str)
args = ap.parse_args()

geom = read_hyp(args.geometry)

add_geom(geom)

events = read_event_list(args.catalog)
pick_dict = get_pick_dict(events)

add_picks(pick_dict)

