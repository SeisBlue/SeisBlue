"""
Logger
"""

import os

import numpy as np

from seisnn import utils


def save_loss(loss_buffer, title, save_dir):
    """
    Write history loss into a log file.

    :type loss_buffer: list
    :param loss_buffer: Loss history.
    :type title: str
    :param title: Log file name.
    :type save_dir: str
    :param save_dir: Output directory.
    """
    utils.make_dirs(save_dir)
    file_path = os.path.join(save_dir, f'{title}.log')
    loss_buffer = np.asarray(loss_buffer)
    with open(file_path, 'ab') as f:
        np.savetxt(f, loss_buffer)


if __name__ == "__main__":
    pass
