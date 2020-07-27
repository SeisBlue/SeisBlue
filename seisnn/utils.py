"""
Utilities
=============

.. autosummary::
    :toctree: utils

    batch
    get_config
    get_dir_list
    make_dirs
    parallel
    parallel_iter
    unet_padding_size

"""

import multiprocessing as mp
import os

import numpy as np
import tqdm
import yaml


def get_config():
    """
    Return path dict in config.yaml.

    :return: config dict
    """
    config_file = os.path.abspath(
        os.path.join(os.path.expanduser("~"), 'config.yaml'))
    with open(config_file, 'r') as file:
        config = yaml.full_load(file)
    return config


def make_dirs(path):
    """
    Create dir if path does not exist.

    :param path:
    """
    if not os.path.isdir(path):
        os.makedirs(path, mode=0o777)


def batch(iterable, n=1):
    """
    Return a batch generator from an iterable and batch size.

    :param iterable:
    :param n:
    """
    iter_len = len(iterable)
    for ndx in range(0, iter_len, n):
        yield iterable[ndx:min(ndx + n, iter_len)]


def parallel(par, file_list):
    """
    Parallelize a partial function and return results in a list.

    :param par:
    :param file_list:
    :return:
    """
    print(f'Parallel in {mp.cpu_count()} threads:')
    batch_size = int(np.ceil(len(file_list) / mp.cpu_count()))
    pool = mp.Pool(processes=mp.cpu_count(), maxtasksperchild=1)
    output = []
    for thread_output in tqdm.tqdm(
            pool.imap_unordered(par, batch(file_list, batch_size)),
            total=int(np.ceil(len(file_list) / batch_size))):
        if thread_output:
            output.extend(thread_output)

    pool.close()
    pool.join()
    return output


def parallel_iter(par, iterator):
    """

    :param par:
    :param iterator:
    :return:
    """
    pool = mp.Pool(processes=mp.cpu_count(), maxtasksperchild=1)
    output = []
    for thread_output in tqdm.tqdm(pool.imap_unordered(par, iterator)):
        if thread_output:
            output.extend(thread_output)

    pool.close()
    pool.join()
    return output


def get_dir_list(file_dir, suffix=""):
    """
    Return directory list from the given path.

    :param file_dir:
    :param suffix:
    :return:
    """
    file_list = []
    for file_name in os.listdir(file_dir):
        f = os.path.join(file_dir, file_name)
        if file_name.endswith(suffix):
            file_list.append(f)

    return file_list


def unet_padding_size(trace, pool_size=2, layers=4):
    """
    Return left and right padding size for a given trace.

    :param trace:
    :param pool_size:
    :param layers:
    :return:
    """
    length = len(trace)
    output = length
    for _ in range(layers):
        output = int(np.ceil(output / pool_size))

    padding = output * (pool_size ** layers) - length
    lpad = 0
    rpad = padding

    return lpad, rpad


if __name__ == "__main__":
    pass
