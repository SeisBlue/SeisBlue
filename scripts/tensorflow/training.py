import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import multi_gpu_model

from seisnn.io import get_dir_list
from seisnn.tensorflow.generator import DataGenerator
from seisnn.tensorflow.model import Nest_Net

pkl_dir = "/mnt/tf_data/pkl/201718select"
pkl_list = get_dir_list(pkl_dir)

split_point = -1000
training_generator = DataGenerator(pkl_list[:split_point], batch_size=2, shuffle=True)
validation_generator = DataGenerator(pkl_list, batch_size=2)

tensorboard = keras.callbacks.TensorBoard(log_dir='../logs', histogram_freq=0,
                                          write_graph=True, write_images=False)
with tf.device('/cpu:0'):
    model = Nest_Net(1, 3001, 1)

model = multi_gpu_model(model, gpus=2)
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights('/mnt/tf_data/weights/pretrained_weight.h5')
model.fit_generator(generator=training_generator, validation_data=validation_generator,
                    epochs=1, use_multiprocessing=True, callbacks=[tensorboard])

model.save_weights('/mnt/tf_data/weights/trained_weight.h5')
