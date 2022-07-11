import os
import numpy as np
from obspy.clients.filesystem import sds
from obspy.core.utcdatetime import UTCDateTime
from seisblue.model.attention import TransformerBlockE, MultiHeadSelfAttention, \
    ResBlock
import tensorflow as tf
import time
import seisblue

import seisblue.example_proto
import seisblue.io
import seisblue.sql
import seisblue.utils


def pre_process(anchor, stream, trace_len, final):
    if anchor < final:

        window_st = stream.slice(starttime=anchor - 5,
                                 endtime=anchor + (trace_len - 1) / 100 + 1)
        window_st.resample(100)
        window_st.detrend('linear')
        window_st.detrend('demean')
        window_st.filter('bandpass', freqmin=1, freqmax=45)

        window_st.trim(anchor, anchor + (trace_len - 1) / 100)
        window_st.normalize()

        all_npts = [item.stats.npts for item in window_st]
        all_npts.sort()
        if all_npts[0] == trace_len:
            instance = seisblue.core.Instance(window_st)
            instance.label = seisblue.core.Label(instance.metadata,
                                                 phase=('P', 'S', 'N'))
            instance.predict = seisblue.core.Label(instance.metadata,
                                                   phase=('P', 'S', 'N'))
            instance.trace.data = instance.trace.data \
                .reshape(-1, trace_len, 3)
            instance.label.data = instance.label.data \
                .reshape(-1, trace_len, 3)
            instance.predict.data = instance.label.data \
                .reshape(-1, trace_len, 3)
            return instance
        else:
            return
    else:
        return


def write_pick(item, trace_len, plot, database):
    item[0].predict.data = item[1]
    if plot:
        item[0].plot(threshold=0.2)
    else:
        item[0].predict.get_picks(threshold=0.2,
                                  from_ms=752,
                                  to_ms=-752)
        for pick in item[0].predict.picks:
            item[0].trace.get_snr(pick)
        item[0].predict.write_picks_to_database('predict',
                                                item[0].metadata.id[:-1],
                                                database)


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    database = 'demo'
    plot = False
    model_instance = '/home/andy/Models/demo_gan.h5'
    config = seisblue.utils.Config()
    db = seisblue.sql.Client(database)
    station_list = db.get_inventories()
    station_list = [station.station for station in station_list]
    station_list.sort()

    model = tf.keras.models.load_model(
        model_instance,
        custom_objects={
            'TransformerBlockE': TransformerBlockE,
            'MultiHeadSelfAttention': MultiHeadSelfAttention,
            'ResBlock': ResBlock
        }
    )
    trace_len = 3008
    client = sds.Client(sds_root=config.sds_root)
    fmtstr = os.path.join(
        "{year}", "{doy:03d}", "{station}",
        "{station}.{network}.{location}.{channel}.{year}.{doy:03d}")
    # client.FMTSTR = fmtstr
    initial_time = UTCDateTime('2020-04-01 00:00:00')
    final = UTCDateTime('2020-04-10 00:00:00')
    for station in station_list:
        print(station)
        start_time = initial_time
        while start_time < final:
            ttt = time.time()
            print(start_time)
            end_time = start_time + 43200
            stream = client.get_waveforms(network="*",
                                          station=station,
                                          location="*",
                                          channel='*',
                                          starttime=start_time,
                                          endtime=end_time
                                          )
            if not stream:
                stream = client.get_waveforms(network="*",
                                              station=station,
                                              location="10",
                                              channel='HH*',
                                              starttime=start_time,
                                              endtime=end_time
                                              )
            if not stream:
                stream = client.get_waveforms(network="*",
                                              station=station,
                                              location="10",
                                              channel='HL*',
                                              starttime=start_time,
                                              endtime=end_time
                                              )
            stream.sort(keys=['channel'], reverse=True)
            stream.merge()
            if stream:
                start_time = stream[0].stats.starttime
                end_time = stream[0].stats.endtime
                shift = np.array(
                    range(
                        int((end_time - start_time) / (
                                (trace_len / 100) - 15.04)))) * (
                                (trace_len / 100) - 15.04)
                anchor_list = [start_time + i for i in shift]
            else:
                anchor_list = []

            if anchor_list:
                instance_list = seisblue.utils.parallel(anchor_list,
                                                        func=pre_process,
                                                        stream=stream,
                                                        trace_len=trace_len,
                                                        final=final)
                a = []
                for item in instance_list:
                    a = a + item
                a = [item for item in a if item is not None]

                instance_input_list = [item.trace.data for item in a]
                start_time = anchor_list[-1] + (trace_len / 100) - 15.04
                if instance_input_list:
                    instance_input_list = np.array(instance_input_list) \
                        .reshape(-1, 1, trace_len, 3)

                    print('predict')
                    predict_output = model.predict(instance_input_list)
                    q = zip(a, predict_output)
                    q = [item for item in q]
                    seisblue.utils.parallel(q,
                                            func=write_pick,
                                            trace_len=trace_len,
                                            plot=plot,
                                            database=database,
                                            )

                print(time.time() - ttt)
            else:
                start_time = start_time + 43200


if __name__ == '__main__':
    main()
