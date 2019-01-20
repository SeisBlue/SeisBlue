import os
import shutil
import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import multi_gpu_model

from obspyNN.io import get_dir_list
from obspyNN.generator import PredictGenerator
from obspyNN.model import Nest_Net
from obspyNN.pick import set_probability

pkl_dir = "/mnt/tf_data/pkl/scan"
pkl_output_dir = pkl_dir + "_predict"
pkl_list = get_dir_list(pkl_dir)

predict_generator = PredictGenerator(pkl_list, batch_size=256)

with tf.device('/cpu:0'):
    model = Nest_Net(1, 3001, 1)

model = multi_gpu_model(model, gpus=2)
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights("/mnt/tf_data/weights/trained_weight.h5")

predict = model.predict_generator(generator=predict_generator,
                                  use_multiprocessing=True, verbose=True)

shutil.rmtree(pkl_output_dir, ignore_errors=True)
os.makedirs(pkl_output_dir, exist_ok=True)
set_probability(predict, pkl_list, pkl_output_dir)
