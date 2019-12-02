import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse

from seisnn.io import read_event_list, write_training_dataset, read_geom
from seisnn.pick import get_pick_list
from seisnn.utils import get_config

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--catalog', required=True, help='catalog s-file', type=str)
ap.add_argument('-g', '--geometry', required=True, help='geometry STATION0.HYP', type=str)
ap.add_argument('-d', '--dataset', required=True, help='output dataset', type=str)
args = ap.parse_args()

config = get_config()

geom = read_geom(args.geometry)

sfile = os.path.join(config['CATALOG_ROOT'], args.catalog)
events = read_event_list(sfile)
pick_list = get_pick_list(events)

write_training_dataset(pick_list, geom, dataset_dir='test', batch_size=100)
