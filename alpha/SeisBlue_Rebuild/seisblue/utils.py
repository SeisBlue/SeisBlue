# -*- coding: utf-8 -*-
import functools
import multiprocessing as mp
import numpy as np
from itertools import chain
import dataclasses
from pathlib import Path
import math
import shutil
import os
import argparse

import seisblue


def batch(iterable, size=1):
    """
    Yields a batch from a list.
    :param iterable: Data list.
    :param int size: Batch size.
    """
    iter_len = len(iterable)
    for ndx in range(0, iter_len, size):
        yield iterable[ndx: min(ndx + size, iter_len)]


def batch_operation(data_list, func, **kwargs):
    """
    Unpacks and repacks a batch.
    :param data_list: List of data.
    :param func: Targeted function.
    :param kwargs: Fixed function parameter.
    :return: List of results.
    """
    return [func(data, **kwargs) for data in data_list]


def _parallel_process(file_list, par, batch_size=None, cpu_count=None,
                      order=False):
    """
    Parallelize a partial function and return results in a list.
    :param list file_list: Process list for partial function.
    :param par: Partial function.
    :rtype: list
    :return: List of results.
    """
    if cpu_count is None:
        cpu_count = mp.cpu_count()
    print(f"Found {cpu_count} cpu threads:")

    pool = mp.Pool(processes=cpu_count, maxtasksperchild=1)

    if not batch_size:
        batch_size = int(np.ceil(len(file_list) / cpu_count))

    if order:
        map_func = pool.map(par, batch(file_list, batch_size))
    else:
        map_func = pool.imap_unordered(par, batch(file_list, batch_size))
    result = list(chain.from_iterable(sublist for sublist in map_func if sublist))
    pool.close()
    pool.join()
    return result


def parallel(data_list, func, batch_size=None, cpu_count=None, order=False,
             iter_attrs=None, **kwargs):
    """
    Parallels a function.
    :param data_list: List of data.
    :param func: Paralleled function.
    :param batch_size:
    :param cpu_count:
    :param kwargs: Fixed function parameters.
    :return: List of results.
    """
    par = functools.partial(batch_operation, func=func, **kwargs)

    result_list = _parallel_process(data_list, par, batch_size, cpu_count,
                                    order)
    return result_list


def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]


def to_dict(obj):
    return {k: v for k, v in dataclasses.asdict(obj).items()}


def check_dir(dirpath, recreate=False, exist_ok=True):
    dir_exists = Path(dirpath).exists()
    if recreate:
        shutil.rmtree(dirpath, ignore_errors=True)
        os.makedirs(dirpath)
    else:
        Path(dirpath).mkdir(parents=True, exist_ok=exist_ok)
    return dir_exists


def calculate_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371
    return round(c * r, 6)


def get_config(congfig_key):
    parser = argparse.ArgumentParser()
    parser.add_argument(congfig_key, type=str, required=True)
    args = parser.parse_args()
    config = seisblue.io.read_yaml(args.data_config_filepath)
    return config


def make_dirs(path):
    """
    Create dir if path does not exist.

    :param str path: Directory path.
    """
    if not os.path.isdir(path):
        os.makedirs(path, mode=0o777, exist_ok=True)