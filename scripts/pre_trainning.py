import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import multi_gpu_model

from obspy import read

import obspyNN
from obspyNN.model import Nest_Net

tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
                                          write_graph=True, write_images=False)

picked_stream = read("/mnt/tf_data/small_set.pkl")
wavefile, probability = obspyNN.io.get_training_set(picked_stream)

split_point = -1000
X_train, X_val = wavefile[0:split_point], wavefile[split_point:]
Y_train, Y_val = probability[0:split_point], probability[split_point:]

with tf.device('/cpu:0'):
    model = Nest_Net(1, 3001, 1)

model = multi_gpu_model(model, gpus=2)
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=2, epochs=1, callbacks=[tensorboard])

evaluate = model.evaluate(X_val, Y_val)
print('Test loss:', evaluate[0])

model.save_weights('pretrained_weight.h5')
