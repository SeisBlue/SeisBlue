import os
import yaml
from os.path import expanduser

from seisnn.utils import make_dirs

SDS_ROOT = '/mnt/SDS_ROOT'
WORKSPACE = os.path.expanduser('~')

TFRECORD_ROOT = os.path.join(WORKSPACE, 'tfrecord')
DATABASE_ROOT = os.path.join(WORKSPACE, 'database')

CATALOG_ROOT = os.path.join(WORKSPACE, 'catalog')
GEOM_ROOT = os.path.join(WORKSPACE, 'geom')
MODELS_ROOT = os.path.join(WORKSPACE, 'models')

config = {'SDS_ROOT': SDS_ROOT,
          'WORKSPACE': WORKSPACE,

          'TFRECORD_ROOT': TFRECORD_ROOT,
          'DATABASE_ROOT': DATABASE_ROOT,

          'CATALOG_ROOT': CATALOG_ROOT,
          'GEOM_ROOT': GEOM_ROOT,
          'MODELS_ROOT': MODELS_ROOT,
          }

# mkdir for all folders and store into config.yaml
if __name__ == '__main__':
    for d in [TFRECORD_ROOT, DATABASE_ROOT, CATALOG_ROOT, MODELS_ROOT, GEOM_ROOT]:
        make_dirs(d)

    with open(os.path.join(expanduser("~"), '.bashrc'), 'w') as file:
        file.write('export PYTHONPATH=/SeisNN:$PYTHONPATH')

    with open('config.yaml', 'w') as file:
        yaml.dump(config, file, sort_keys=False)

    print('SeisNN initialized.')
