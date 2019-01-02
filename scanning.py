from obspy import UTCDateTime
import numpy as np

from obspyNN.model import unet
import obspyNN

sds_root = "/mnt/Data"

model = unet()
model.load_weights("/mnt/tf_data/trained_weight.h5")


start_time = UTCDateTime("2017-05-01 00:00:00")
end_time = UTCDateTime("2017-05-01 01:00:00")

nslc = ("*", "H007", "*", "??Z")
stream = obspyNN.io.scan_station(sds_root, nslc, start_time, end_time)

wavefile = []
for trace in stream:
    wavefile.append(trace.data)

output_shape = (stream.count(), 1, 3001, 1)
wavefile = np.asarray(wavefile).reshape(output_shape)

predict = model.predict(wavefile)
result = obspyNN.probability.set_probability(stream, predict)
obspyNN.plot.plot_stream(stream)
