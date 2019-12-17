import os
from seisnn.utils import make_dirs, get_config

config = get_config()

def save_history(feature, title, save_dir):
    make_dirs(save_dir)
    file_path = os.path.join(save_dir, f'{title}.tfrecord')
    feature.to_tfrecord(file_path)