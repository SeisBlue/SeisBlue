import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import multi_gpu_model
from obspy import read
import obspyNN
from obspyNN.model import Nest_Net

picked_stream = read("/mnt/tf_data/pkl/small_set.pkl")
wavefile, probability = obspyNN.io.get_training_set(picked_stream)

with tf.device('/cpu:0'):
    model = Nest_Net(1, 3001, 1)

model = multi_gpu_model(model, gpus=2)
model.compile(optimizer=Adam(lr=3e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights("/mnt/tf_data/weights/trained_weight.h5")

predict = model.predict(wavefile, batch_size=2, verbose=True)
result = obspyNN.probability.set_probability(picked_stream, predict)
result.write("/mnt/tf_data/pkl/predict_small_set.pkl", format="PICKLE")
