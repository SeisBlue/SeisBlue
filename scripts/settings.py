import os
from seisnn.io import make_dirs
"""
workspace/
├── tfrecord/
│   ├── dataset/
│   │   └── dataset_foo/
│   │       └── 2017-04-25T23:14:27.995000HL.H009.XX.EPZ.tfrecord
│   └── catalog/
│       └── 2017select.tfrecord
│       
└── models/
    └── model_foo/
        ├── weight/
        │   └── pretrained_weight.h5
        ├── logs/
        └── plot/
"""
SDS_ROOT = '/mnt/SDS_ROOT'
WORKSPACE = '/home/jimmy'

TFRECORD_ROOT = os.path.join(WORKSPACE, 'tfrecord')
DATASET_ROOT = os.path.join(TFRECORD_ROOT, 'dataset')
PICK_ROOT = os.path.join(TFRECORD_ROOT, 'catalog')

MODELS_ROOT = os.path.join(WORKSPACE, 'models')

# mkdir for all folders
if __name__ == '__main__':
    for d in [TFRECORD_ROOT, DATASET_ROOT, PICK_ROOT, MODELS_ROOT]:
        make_dirs(d)