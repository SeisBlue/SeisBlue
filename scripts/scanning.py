import tensorflow as tf
from obspy import UTCDateTime
import numpy as np
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras.optimizers import Adam
from obspyNN.model import Nest_Net
import obspyNN

sds_root = "/mnt/DATA"

with tf.device('/cpu:0'):
    model = Nest_Net(1, 3001, 1)

model = multi_gpu_model(model, gpus=2)
model.compile(optimizer=Adam(lr=3e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights("/mnt/tf_data/weights/trained_weight.h5")

start_time = UTCDateTime("2018-02-09 12:47:54")
end_time = start_time + 30

nslc = ("HL", "*", "*", "??Z")
stream = obspyNN.io.scan_station(sds_root, nslc, start_time, end_time)

wavefile = []
for trace in stream:
    wavefile.append(trace.data)

output_shape = (stream.count(), 1, 3001, 1)
wavefile = np.asarray(wavefile).reshape(output_shape)

predict = model.predict(wavefile, batch_size=2, verbose=True)
result = obspyNN.probability.set_probability(stream, predict)
for trace in result:
    obspyNN.plot.plot_trace(trace)
