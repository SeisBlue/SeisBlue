import obspy
import numpy as np
import seisnn
from tensorflow.keras.utils import Progbar
import os

CVA_list = seisnn.utils.get_dir_list('/home/andy/CWB/20*/',
                                     suffix='.txt')
progbar = Progbar(len(CVA_list))
for q, list in enumerate(CVA_list):
    f = open(list, 'r')
    lines = f.readlines()
    count = 0
    t = []
    e = []
    n = []
    z = []
    filename = str(list[-24:-4]).replace('/', '')
    if os.path.exists(f'/home/andy/mseed/CVA_TO_MSEED/{filename}.mseed'):
        progbar.add(1)
        continue
    try:
        for line in lines:
            if count == 0:
                station_code = line.strip()[14:20]
                if len(station_code) > 4:
                    station_code = seisnn.io.change_station_code(station_code)
                    network = 'TSMIP'
                else:
                    network = 'CWBSN'
            elif count == 2:
                start_time = obspy.UTCDateTime(line.strip()[12:-1])
            elif count == 3:
                duration = float(line.strip()[20:28])
            elif count == 4:
                samplerate = int(line.strip()[17:20])
            elif count == 1:
                pass
            elif count == 5:
                pass
            elif count == 6:
                pass
            elif count == 7:
                pass
            elif count == 8:
                pass
            elif count == 9:
                pass
            elif count == 10:
                pass
            else:
                try:
                    tt = float(line[0:10])
                except ValueError:
                    tt = 0
                try:
                    te = float(line[10:20])
                except ValueError:
                    te = 0
                try:
                    tn = float(line[20:30])
                except ValueError:
                    tn = 0
                try:
                    tz = float(line[30:40])
                except ValueError:
                    tz = 0
                t.append(tt)
                e.append(te)
                n.append(tn)
                z.append(tz)
            count = count + 1

        traceE = obspy.core.trace.Trace(np.array(e))
        traceN = obspy.core.trace.Trace(np.array(n))
        traceZ = obspy.core.trace.Trace(np.array(z))

        for i, trace in enumerate([traceE, traceN, traceZ]):
            try:
                trace.stats.network = network
                trace.stats.station = station_code
                trace.stats.starttime = start_time
                trace.stats.sampling_rate = samplerate
                trace.stats.npts = len(t)
                trace.stats.delta = t[1] - t[0]
            except IndexError:
                print(list)
            if i == 0:
                trace.stats.channel = 'EHE'
            if i == 1:
                trace.stats.channel = 'EHN'
            if i == 2:
                trace.stats.channel = 'EHZ'

        st = obspy.core.stream.Stream([traceE, traceN, traceZ])
        progbar.add(1)

        st.write(f'/home/andy/mseed/CVA_TO_MSEED/{filename}.mseed',
                 format='MSEED')
        f.close()
    except ValueError:
        print(list)
