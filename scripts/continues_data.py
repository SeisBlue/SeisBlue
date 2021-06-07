import os

import obspy
import seisnn
import tensorflow as tf
import numpy as np
from seisnn.model.attention import TransformerBlockE, TransformerBlockD, \
    MultiHeadSelfAttention, ResBlock

database = 'Hualien.db'
db = seisnn.sql.Client(database)
model = tf.keras.models.load_model(
    '/home/andy/Models/CWB_trans_2010_2019.h5',
    custom_objects={
        'TransformerBlockE': TransformerBlockE,
        'TransformerBlockD': TransformerBlockD,
        'MultiHeadSelfAttention': MultiHeadSelfAttention,
        'ResBlock': ResBlock
    })
a =  obspy.UTCDateTime(0)
year = 2019
tfr_converter = seisnn.components.TFRecordConverter(trace_length=86401)
station_list = os.listdir(f'/home/andy/SDS_ROOT/{year}/HL/')
station_list.sort()
for station in station_list:
    julday = seisnn.utils.get_dir_list(f'/home/andy/SDS_ROOT/{year}/HL/{station}', suffix='2019*')
    julday.sort()
    anchor_time = obspy.UTCDateTime(year = year,julday=julday[0][-3:])
    print(station)
    station_list = os.path.basename(f'/home/andy/SDS_ROOT/2019/HL/{station}/')
    while True:
        if anchor_time == a:

            break
        a = anchor_time
        print(anchor_time)
        metadata = tfr_converter.get_time_window(anchor_time=anchor_time, station = station,shift=0 )
        streams = seisnn.io.read_sds(metadata,trim=False)
        for _, stream in streams.items():
            print(f'stream processing')
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
                instance.plot(threshold = 0.4)
                # instance.predict.get_picks()
                # instance.predict.write_picks_to_database('predict', database)

db.remove_duplicates('pick', ['time', 'phase', 'station', 'tag'])
