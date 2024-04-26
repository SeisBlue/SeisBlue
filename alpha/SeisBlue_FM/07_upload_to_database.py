# -*- coding: utf-8 -*-
import os
import subprocess
import warnings
import glob
import obspy.io.nordic.core
import re
import shutil
from tqdm import tqdm

from seisblue import tool, io

if __name__ == '__main__':
    config = io.read_yaml('./config/data_config.yaml')
    c = config['global']

    result_dir = os.path.join('./result', c['dataset_name'], 'fine')
    obspy_events = io.get_obspy_events(c['events_dir'])
