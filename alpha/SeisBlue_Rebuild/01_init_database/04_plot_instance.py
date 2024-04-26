# -*- coding: utf-8 -*-
import seisblue

import argparse
import h5py
import glob
from itertools import chain


def get_dataset(filename):
    instances = []
    with h5py.File(filename, 'r') as f:
        for id, instance_h5 in f.items():
            instance = seisblue.io.read_hdf5(instance_h5)
            instances.append(instance)
    return instances


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config_filepath", type=str, required=True)
    args = parser.parse_args()
    config = seisblue.io.read_yaml(args.data_config_filepath)

    filepaths = list(sorted(glob.glob('/usr/src/app/dataset/HP*')))
    instances = seisblue.utils.parallel(filepaths,
                                            func=get_dataset)
    instances = list(chain.from_iterable(
        sublist for sublist in instances if sublist))
    fig_dir = f'/usr/src/app/01_init_database/figure/check_instance'
    seisblue.utils.check_dir(fig_dir, recreate=True)
    for instance in instances[11:30]:
        id = instance.timewindow.starttime
        seisblue.plot.plot_dataset(instance, save_dir=fig_dir, title=f'{id}')
