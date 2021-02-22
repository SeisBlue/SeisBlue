"""
Logger
"""

import os

import numpy as np

import seisnn.utils


def save_loss(loss_buffer, title, save_dir):
    """
    Write history loss into a log file.

    :param loss_buffer: Loss history.
    :param str title: Log file name.
    :param str save_dir: Output directory.
    """
    seisnn.utils.make_dirs(save_dir)
    file_path = os.path.join(save_dir, f'{title}.log')
    loss_buffer = np.asarray(loss_buffer)
    with open(file_path, 'ab') as f:
        np.savetxt(f, loss_buffer)


if __name__ == "__main__":
    pass
