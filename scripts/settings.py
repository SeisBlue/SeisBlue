import os

SDS_ROOT = '/mnt/SDS_ROOT'
WORKSPACE = '/home/jimmy'
TFRECORD_ROOT = os.path.join(WORKSPACE, 'tfrecord')
MODELS_ROOT = os.path.join(WORKSPACE, 'models')

# initialize workspace
if __name__ == '__main__':
    os.mkdir()