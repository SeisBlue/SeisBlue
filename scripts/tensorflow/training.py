import os

from tensorflow.python import keras
from tensorflow.python.keras.optimizers import Adam

from seisnn.io import get_dir_list
from seisnn.tensorflow.generator import DataGenerator
from seisnn.tensorflow.model import Nest_Net

pkl_dir = "/mnt/tf_data/pkl/201718select"
pkl_list = get_dir_list(pkl_dir)

split_point = -1000

training_generator = DataGenerator(pkl_list[:split_point], batch_size=2, shuffle=False)
validation_generator = DataGenerator(pkl_list[split_point:], batch_size=32)

tensorboard = keras.callbacks.TensorBoard(log_dir='../logs', histogram_freq=0,
                                          write_graph=True, write_images=False)

model = Nest_Net(1, 3001, 1)
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights('/mnt/tf_data/weights/pretrained_weight.h5')
model.fit_generator(generator=training_generator, validation_data=validation_generator,
                    epochs=1, use_multiprocessing=True, callbacks=[tensorboard])

weight_dir = "/mnt/tf_data/weights"
os.makedirs(weight_dir, exist_ok=True)
model.save_weights('/mnt/tf_data/weights/trained_weight.h5')