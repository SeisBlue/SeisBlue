"""
Utilities
"""

import functools
import multiprocessing as mp
import numpy as np


def parallel(data_list, func, batch_size=None, cpu_count=None, **kwargs):
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

    result_list = _parallel_process(data_list, par, batch_size, cpu_count)
    return result_list


def batch_operation(data_list, func, **kwargs):
    """
    Unpacks and repacks a batch.

    :param data_list: List of data.
    :param func: Targeted function.
    :param kwargs: Fixed function parameter.

    :return: List of results.
    """
    return [func(data, **kwargs) for data in data_list]


def _parallel_process(file_list, par, batch_size=None, cpu_count=None):
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
    map_func = pool.imap_unordered(par, batch(file_list, batch_size))
    result = [output for output in map_func]

    pool.close()
    pool.join()
    return result


def batch(iterable, size=1):
    """
    Yields a batch from a list.

    :param iterable: Data list.
    :param int size: Batch size.
    """
    iter_len = len(iterable)
    for ndx in range(0, iter_len, size):
        yield iterable[ndx : min(ndx + size, iter_len)]