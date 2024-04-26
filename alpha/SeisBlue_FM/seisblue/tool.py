# -*- coding: utf-8 -*-
import os
import dataclasses
import functools
import multiprocessing as mp
import numpy as np
import argparse
import shutil
from pathlib import Path


def to_dict(obj):
    return {k: v for k, v in dataclasses.asdict(obj).items()}


def evaluate_string(value):
    try:
        return eval(value)
    except Exception as e:
        return value


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
    result = [item for sublist in map_func for item in sublist]

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=pathlib.Path)
    return parser.parse_args()


def check_dir(dirpath, recreate=False, exist_ok=True):
    dir_exists = Path(dirpath).exists()
    if recreate:
        shutil.rmtree(dirpath, ignore_errors=True)
        os.makedirs(dirpath)
    else:
        Path(dirpath).mkdir(parents=True, exist_ok=exist_ok)
    return dir_exists