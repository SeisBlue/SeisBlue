from obspy import UTCDateTime
import numpy as np
from tensorflow.python.keras.optimizers import Adam, RMSprop, SGD
from obspyNN.model import U_Net
import obspyNN

sds_root = "/mnt/Data"

model = U_Net(1, 3001, 1)
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights("/mnt/tf_data/trained_weight.h5")


start_time = UTCDateTime("2017-03-19 23:00:00")
end_time = start_time + 3600

nslc = ("HL", "H065", "*", "??Z")
stream = obspyNN.io.scan_station(sds_root, nslc, start_time, end_time)

wavefile = []
for trace in stream:
    wavefile.append(trace.data)

output_shape = (stream.count(), 1, 3001, 1)
wavefile = np.asarray(wavefile).reshape(output_shape)

predict = model.predict(wavefile)
result = obspyNN.probability.set_probability(stream, predict)
obspyNN.plot.plot_stream(stream)
