"""
Utilities
"""

import functools
import glob
import multiprocessing as mp
import os

import numpy as np
import tqdm
import yaml


class Config:
    __slots__ = [
        'workspace',
        'sds_root',
        'sfile_root',

        'tfrecord',
        'train',
        'test',
        'eval',

        'sql_database',
        'catalog',
        'geom',
        'models',
    ]

    def __init__(self, workspace=os.path.expanduser("~"), initialize=False):
        if initialize:
            self.generate_config()
            self.load_config(workspace)
            self.create_folders()

        try:
            self.load_config(workspace)
        except FileNotFoundError:
            print('Missing config.yml, please use Config(initialize=True).')

    def load_config(self, workspace=os.path.expanduser("~")):
        config_file = os.path.abspath(os.path.join(workspace, 'config.yml'))

        with open(config_file, 'r') as file:
            config = yaml.full_load(file)
            self.workspace = config['WORKSPACE']
            self.sds_root = config['SDS_ROOT']
            self.sfile_root = config['SFILE_ROOT']

            self.tfrecord = config['TFRecord']
            self.train = config['Train']
            self.test = config['Test']
            self.eval = config['Eval']

            self.sql_database = config['SQL_Database']
            self.catalog = config['Catalog']
            self.geom = config['Geom']
            self.models = config['Models']

    @staticmethod
    def generate_config(workspace=os.path.expanduser('~')):
        config = {
            'WORKSPACE': workspace,
            'SDS_ROOT': os.path.join(workspace, 'SDS_ROOT'),
            'SFILE_ROOT': os.path.join(workspace, 'SFILE_ROOT'),

            'TFRecord': os.path.join(workspace, 'TFRecord'),
            'Train': os.path.join(workspace, 'TFRecord', 'Train'),
            'Test': os.path.join(workspace, 'TFRecord', 'Test'),
            'Eval': os.path.join(workspace, 'TFRecord', 'Eval'),

            'SQL_Database': os.path.join(workspace, 'SQL_Database'),
            'Catalog': os.path.join(workspace, 'Catalog'),
            'Geom': os.path.join(workspace, 'Geom'),
            'Models': os.path.join(workspace, 'Models'),
        }

        path = os.path.join(workspace, 'config.yml')
        with open(path, 'w') as file:
            yaml.dump(config, file, sort_keys=False)
        print(f'Create config: {path}')

    def create_folders(self):
        path_list = [
            self.tfrecord,
            self.train,
            self.test,
            self.eval,
            self.sql_database,

            self.catalog,
            self.geom,
            self.models,
        ]

        for d in path_list:
            make_dirs(d)
            print(f'Create folder: {d}')


def make_dirs(path):
    """
    Create dir if path does not exist.

    :param str path: Directory path.
    """
    if not os.path.isdir(path):
        os.makedirs(path, mode=0o777)


def batch(iterable, size=1):
    """
    Yields a batch from a list.

    :param iterable: Data list.
    :param int size: Batch size.
    """
    iter_len = len(iterable)
    for ndx in range(0, iter_len, size):
        yield iterable[ndx:min(ndx + size, iter_len)]


def batch_operation(data_list, func, **kwargs):
    """
    Unpacks and repacks a batch.

    :param data_list: List of data.
    :param func: Targeted function.
    :param kwargs: Fixed function parameter.

    :return: List of results.
    """
    return [func(data, **kwargs) for data in data_list]


def _parallel_process(file_list, par, batch_size=None):
    """
    Parallelize a partial function and return results in a list.

    :param list file_list: Process list for partial function.
    :param par: Partial function.
    :rtype: list

    :return: List of results.
    """
    cpu_count = mp.cpu_count()
    print(f'Found {cpu_count} cpu threads:')

    pool = mp.Pool(processes=cpu_count, maxtasksperchild=1)

    if not batch_size:
        batch_size = int(np.ceil(len(file_list) / cpu_count))
    map_func = pool.imap_unordered(par, batch(file_list, batch_size))
    result = [output for output in map_func]

    pool.close()
    pool.join()
    return result


def parallel(data_list, func, batch_size=None, **kwargs):
    """
    Parallels a function.

    :param data_list: List of data.
    :param func: Paralleled function.
    :param kwargs: Fixed function parameters.

    :return: List of results.
    """
    par = functools.partial(batch_operation, func=func, **kwargs)

    result_list = _parallel_process(data_list, par, batch_size)
    return result_list


def _parallel_iter(par, iterator):
    """
    Parallelize a partial function and return results in a list.

    :param par: Partial function.
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


def get_dir_list(file_dir, suffix="", recursive=True):
    """
    Returns directory list from the given path.

    :param str file_dir: Target directory.
    :param str suffix: (Optional.) File extension, Ex: '.tfrecord'.
    :param bool recursive: (Optional.) Search directory recursively. Default is True.
    :rtype: list
    :return: List of file name.
    """
    file = os.path.join(file_dir, f'**/*{suffix}')
    file_list = glob.glob(file, recursive=recursive)
    file_list = sorted(file_list)

    return file_list


def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]


def unet_padding_size(trace, pool_size=2, layers=4):
    """
    Return left and right padding size for a given trace.

    :param np.array trace: Trace array.
    :param int pool_size: (Optional.) Unet pool size, default is 2.
    :param int layers: (Optional.) Unet stages, default is 4.
    :return: (left padding size, right padding size)
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
