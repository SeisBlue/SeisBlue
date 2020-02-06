import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
from tqdm import tqdm
from multiprocessing import cpu_count
import tensorflow as tf

from seisnn.io import read_event_list, write_training_dataset, read_hyp
from seisnn.pick import get_pick_dict
from seisnn.utils import get_config

print(f'cpu counts: {cpu_count()} threads')

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--catalog', required=True, help='catalog s-file dir', type=str)
ap.add_argument('-g', '--geometry', required=True, help='geometry STATION0.HYP', type=str)
ap.add_argument('-d', '--dataset', required=True, help='output dataset name', type=str)
ap.add_argument('-p', '--pickset', required=True, help='output pickset name', type=str)

args = ap.parse_args()
config = get_config()

geom = read_hyp(args.geometry)
events = read_event_list(args.catalog)
pick_dict = get_pick_dict(events)
pick_dict_keys = pick_dict.keys()
for i, key in enumerate(pick_dict_keys):
    tqdm.write(f'station {key}, total: {i + 1}/{len(pick_dict_keys)}, pick counts: {len(pick_dict[key])}')
    with tf.device('/cpu:0'):
        write_training_dataset(pick_dict[key], geom, dataset=args.dataset, pickset=args.pickset)
