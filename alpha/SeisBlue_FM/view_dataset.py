# -*- coding: utf-8 -*-
import seisblue.io
from seisblue import tool, plot, io
import random
import h5py
import glob
import os
import subprocess
import shutil
import matplotlib.pyplot as plt
from pathlib import Path


def get_evt_instance(filepath):
    with h5py.File(filepath, "r") as f:
        event_index = random.randint(0, len(f.values()) - 1)
        event = list(f.values())[event_index]
        instance_index = random.randint(0, len(event['instances'].values()) - 1)
        event = seisblue.io.read_hdf5_instance(event, instance_index)

    return event.instances


def use_hypocenter(temp_filename, update=False):
    if update:
        hypo_result = subprocess.run(
            ['hyp', temp_filename, '-update'],
            input=b'y',
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    else:
        hypo_result = subprocess.run(
            ['hyp', temp_filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    return hypo_result


def focmec_o(filename):
    result = subprocess.run(
        ['focmec', 'o', filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    print(result)

if __name__ == '__main__':
    config = seisblue.io.read_yaml('./config/data_config.yaml')
    c = config['global']
    # file = glob.glob(f'./dataset/2023_PL*test*.hdf5')[0]
    # instances = get_evt_instance(file)
    # plot.plot_dataset(instances[0], title="2022_ETBATS", save_dir=f'./figure/{c["dataset_name"]}')

    # events_dir = glob.glob(f"./result/{c['dataset_name']}/fine/best")[0]
    # cwd = os.getcwd()
    # os.chdir(events_dir)
    # shutil.copy(c['hyp_filepath'], '.')
    # shutil.copy(c['def_filepath'], '.')
    #
    # use_hypocenter('31-2001-14L.S202305')
    # focmec_o('31-2001-14L.S202305')
    # os.chdir(cwd)


