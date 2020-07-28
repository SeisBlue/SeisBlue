"""
Utilities
"""

import multiprocessing as mp
import os

import numpy as np
import tqdm
import yaml


def get_config():
    """
    Returns path dict in config.yaml.

    :rtype: dict
    :return: Dictionary contains predefined workspace paths.
    """
    config_file = os.path.abspath(
        os.path.join(os.path.expanduser("~"), 'config.yaml'))
    with open(config_file, 'r') as file:
        config = yaml.full_load(file)
    return config


def make_dirs(path):
    """
    Create dir if path does not exist.

    :type path: str
    :param path: Directory path.
    """
    if not os.path.isdir(path):
        os.makedirs(path, mode=0o777)


def batch(iterable, n=1):
    """
    Yield a batch from a list.

    :type iterable: list
    :param iterable: Data list.
    :type n: int
    :param n: Batch size.
    """
    iter_len = len(iterable)
    for ndx in range(0, iter_len, n):
        yield iterable[ndx:min(ndx + n, iter_len)]


def parallel(par, file_list):
    """
    Parallelize a partial function and return results in a list.

    :type par: object
    :param par: Partial function.
    :type file_list: list
    :param file_list: Process list for partial function.
    :rtype: list
    :return: List of results.
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
    Parallelize a partial function and return results in a list.

    :type par: any
    :param par: Partial function.
    :type iterator: any
    :param iterator: Iterable object.
    :rtype: list
    :return: List of results.
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
    Returns directory list from the given path.

    :type file_dir: str
    :param file_dir: Target directory.
    :type suffix: str
    :param suffix: (Optional.) File extension.
    :rtype: list
    :return: List of file name.
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
