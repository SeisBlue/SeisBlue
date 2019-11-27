import os

from obspy import read_events
from seisnn.io import read_event_list, write_training_dataset, get_pick_list
from seisnn.utils import get_config

config = get_config()

sfile_list = [os.path.join(config['PICK_ROOT'], "201718select.out")]
output_dir = os.path.join(config['DATASET_ROOT'], "201718select_random")
pick_tfrecord = os.path.join(config['PICK_ROOT'], "saved_catalog.tfrecord")

catalog = read_event_list(sfile_list)

write_training_dataset(catalog, config['SDS_ROOT'], output_dir=output_dir, batch_size=100, remove_dir=True)
