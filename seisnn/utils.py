"""
Utilities
=============

.. autosummary::
    :toctree: utils

    batch
    binary_search
    get_config
    get_dir_list
    make_dirs
    parallel
    parallel_iter
    unet_padding_size

"""

import os
from bisect import bisect_left, bisect_right
from multiprocessing import Pool, cpu_count

import numpy as np
import yaml
from tqdm import tqdm


def get_config():
    config_file = os.path.abspath(os.path.join(os.path.expanduser("~"), 'config.yaml'))
    with open(config_file, 'r') as file:
        config = yaml.full_load(file)
    return config


def make_dirs(path):
    if not os.path.isdir(path):
        os.makedirs(path, mode=0o777)


def batch(iterable, n=1):
    iter_len = len(iterable)
    for ndx in range(0, iter_len, n):
        yield iterable[ndx:min(ndx + n, iter_len)]


def parallel(par, file_list):
    batch_size = int(np.ceil(len(file_list) / cpu_count()))
    pool = Pool(processes=cpu_count(), maxtasksperchild=1)
    output = []
    for thread_output in tqdm(pool.imap_unordered(par, batch(file_list, batch_size)),
                              total=int(np.ceil(len(file_list) / batch_size))):
        if thread_output:
            output.extend(thread_output)

    pool.close()
    pool.join()
    return output

def parallel_iter(par, iterator):
    pool = Pool(processes=cpu_count(), maxtasksperchild=1)
    output = []
    for thread_output in tqdm(pool.imap_unordered(par, iterator)):
        if thread_output:
            output.extend(thread_output)

    pool.close()
    pool.join()
    return output


def get_dir_list(file_dir, suffix=""):
    file_list = []
    for file_name in os.listdir(file_dir):
        f = os.path.join(file_dir, file_name)
        if file_name.endswith(suffix):
            file_list.append(f)

    return file_list


def unet_padding_size(trace, pool_size=2, layers=4):
    length = len(trace)
    output = length
    for _ in range(layers):
        output = int(np.ceil(output / pool_size))

    padding = output * (pool_size ** layers) - length
    lpad = 0
    rpad = padding

    return lpad, rpad


def binary_search(key_list, min_value, max_value):
    # binary search, key_list must be sorted by time
    left = bisect_left(key_list, min_value)
    right = bisect_right(key_list, max_value)
    return left, right

if __name__ == "__main__":
    pass
