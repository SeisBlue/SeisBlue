import os
from multiprocessing import Pool, cpu_count

import numpy as np
import yaml
from tqdm import tqdm


def get_config():
    config_file = os.path.join(os.path.expanduser('~'), 'SeisNN', 'config.yaml')
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


def parallel(par, file_list, batch_size=100):
    pool = Pool(processes=cpu_count(), maxtasksperchild=1)

    for _ in tqdm(pool.imap_unordered(par, batch(file_list, batch_size)),
                  total=int(np.ceil(len(file_list) / batch_size))):
        pass

    pool.close()
    pool.join()


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
    lpad = int(np.ceil(padding / 2))
    rpad = int(np.floor(padding / 2))

    return lpad, rpad