# -*- coding: utf-8 -*-
import os
import glob

import seisblue.io
from seisblue import tool, io


def main():
    config = seisblue.io.read_yaml('./config/data_config.yaml')
    c = config['global']
    config = seisblue.io.read_yaml('./config/model_config.yaml')
    c_model = config['test']

    filepaths = glob.glob(f'./dataset/{c["dataset_name"]}*test*.hdf5')
    result_dir = os.path.join('./result', c["dataset_name"])
    tool.check_dir(result_dir, recreate=True)
    tool.parallel(filepaths,
                  func=io.hdf5_to_sfile,
                  out_dir=result_dir,
                  method_id=c_model['tag'],
                  flip_polarity=c_model['reverse_back'])


if __name__ == '__main__':
    main()
