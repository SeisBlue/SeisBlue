from obspy import UTCDateTime
import numpy as np
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras.optimizers import Adam
from obspyNN.model import U_Net, Nest_Net
import obspyNN

sds_root = "/mnt/DATA"

model = Nest_Net(1, 3001, 1)
parallel_model = multi_gpu_model(model, gpus=2)
parallel_model.compile(optimizer=Adam(lr=3e-4), loss='binary_crossentropy', metrics=['accuracy'])
parallel_model.load_weights("/mnt/tf_data/nest_net_trained_weight.h5")


start_time = UTCDateTime("2018-02-10 21:20:22")
end_time = start_time + 30

nslc = ("HL", "*", "*", "??Z")
stream = obspyNN.io.scan_station(sds_root, nslc, start_time, end_time)

wavefile = []
for trace in stream:
    wavefile.append(trace.data)

output_shape = (stream.count(), 1, 3001, 1)
wavefile = np.asarray(wavefile).reshape(output_shape)

predict = parallel_model.predict(wavefile)
result = obspyNN.probability.set_probability(stream, predict)
obspyNN.plot.plot_trace(stream)
