import obspy
import seisnn
import tensorflow as tf
import numpy as np
from seisnn.model.attention import TransformerBlockE, TransformerBlockD, \
    MultiHeadSelfAttention, ResBlock

model_path = '/home/andy/Models/CWB_trans_2010_2019.h5'
year = 2019
month = 4
day = 20
anchor_time = obspy.UTCDateTime(f'{year}-{month}-{day}')
model = tf.keras.models.load_model(
    model_path,
    custom_objects={
        'TransformerBlockE': TransformerBlockE,
        'TransformerBlockD': TransformerBlockD,
        'MultiHeadSelfAttention': MultiHeadSelfAttention,
        'ResBlock': ResBlock
    })

tfr_converter = seisnn.components.TFRecordConverter(trace_length=15001)
count = 0
while True:
    flag = 0
    metadata = tfr_converter.get_time_window(anchor_time=anchor_time, station='HL901', shift=0)
    streams = seisnn.io.read_sds(metadata)
    for _, stream in streams.items():
        print(f'stream processing anchor_time = {anchor_time})')
        stream.resample(100)
        stream.merge()
        print('start sliding')
        instance_list = []
        instance_input_list = []
        for window_st in stream.slide(window_length=30.07, step=30.00):

            window_st.normalize()
            if window_st.traces[0].stats.npts != 3008:
               continue
            instance = seisnn.core.Instance(window_st)
            instance.label = seisnn.core.Label(instance.metadata,
                                               tfr_converter.phase)
            instance.predict = seisnn.core.Label(instance.metadata,
                                                 tfr_converter.phase)
            instance_input_list.append(instance.trace.data.reshape(1,3008,3))
            instance.trace.data = instance.trace.data.reshape(-1, 3008, 3)
            instance.label.data = instance.label.data.reshape(-1, 3008, 3)

            instance_list.append(instance)
            anchor_time = window_st.traces[0].stats.starttime
        print('predict')
        instance_input_list = np.array(instance_input_list)
        predict_output = model.predict(instance_input_list)
        for i,instance in enumerate(instance_list):
            instance.predict.data = predict_output[i]
            # instance.plot()