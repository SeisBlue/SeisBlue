import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse

from seisnn.io import read_event_list, write_training_dataset
from seisnn.utils import get_config

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--catalog', required=True, help='catalog s-file', type=str)
ap.add_argument('-d', '--dataset', required=True, help='output dataset', type=str)

args = ap.parse_args()

config = get_config()

sfile_list = [os.path.join(config['CATALOG_ROOT'], args.catalog)]
catalog = read_event_list(sfile_list)

write_training_dataset(catalog,  dataset_dir='test', batch_size=100)
