"""
Logger
=============

.. autosummary::
   :toctree: logger

"""

import os
import numpy as np
from seisnn.utils import make_dirs, get_config

config = get_config()


def save_loss(loss_buffer, title, save_dir):
    make_dirs(save_dir)
    file_path = os.path.join(save_dir, f'{title}.log')
    loss_buffer = np.asarray(loss_buffer)
    with open(file_path, 'ab') as f:
        np.savetxt(f, loss_buffer)

if __name__ == "__main__":
    pass
