import seisblue
from datetime import datetime
from obspy import UTCDateTime
import numpy as np
from seisblue.associator.sql import AssociatedEvent
from seisblue.plot import plot_event_residual
from seisblue.sql import Event, Pick


def get_ot_from_time_to_time(time, delta=0.1):
    from_time = UTCDateTime(time) - delta
    from_time = datetime.strptime(str(from_time), '%Y-%m-%dT%H:%M:%S.%fZ')
    to_time = UTCDateTime(time) + delta
    to_time = datetime.strptime(str(to_time), '%Y-%m-%dT%H:%M:%S.%fZ')
    return from_time, to_time


def compare(station, database, from_time=None, to_time=None, low_confidence=0.2,
            low_snr=0,
            delta=0.2, residual_list=False):
    db = seisblue.sql.Client(database=database)
    P_tp = 0
    S_tp = 0
    p_residual = []
    s_residual = []
    P_picks = db.get_picks(tag='manual',
                           phase='P',
                           station=station.station,

                           from_time=from_time,
                           to_time=to_time)
    S_picks = db.get_picks(tag='manual',
                           phase='S',
                           station=station.station,

                           from_time=from_time,
                           to_time=to_time)

    P_predict_pick = db.get_picks(from_time=from_time,
                                  to_time=to_time,
                                  tag='predict',
                                  phase='P',
                                  station=station.station,
                                  low_confidence=low_confidence,
                                  # low_snr=low_snr,
                                  flatten=False)
    S_predict_pick = db.get_picks(from_time=from_time,
                                  to_time=to_time,
                                  tag='predict',
                                  phase='S',
                                  station=station.station,
                                  low_confidence=low_confidence,
                                  # low_snr=low_snr,
                                  flatten=False)
    for pick in P_picks:
        from_time, to_time = seisblue.compare.get_ot_from_time_to_time(
            pick.time, delta)
        correspond_pick = P_predict_pick.filter(Pick.time >= from_time)
        correspond_pick = correspond_pick.filter(Pick.time <= to_time)
        correspond_pick = correspond_pick.all()
        if correspond_pick:
            P_tp = P_tp + 1
            if residual_list:
                residual = pick.time - correspond_pick[0].time
                p_residual.append(residual.total_seconds())

    for pick in S_picks:
        from_time, to_time = seisblue.compare.get_ot_from_time_to_time(
            pick.time, delta)
        correspond_pick = S_predict_pick.filter(Pick.time >= from_time)
        correspond_pick = correspond_pick.filter(Pick.time <= to_time)
        correspond_pick = correspond_pick.all()
        if correspond_pick:
            S_tp = S_tp + 1
            if residual_list:
                residual = pick.time - correspond_pick[0].time
                s_residual.append(residual.total_seconds())
    tp = [P_tp, S_tp]
    residual = [p_residual, s_residual]
    print(station.station, tp)
    return [tp, residual]


def compare_dataset_pick(database, from_time=None, to_time=None, station=None,
                         low_confidence=None, low_snr=None,delta=0.2,
                         residual_list=False):
    db = seisblue.sql.Client(database=database)

    stations = db.get_inventories(station=station)
    tp = seisblue.utils.parallel(stations,
                                           func=compare,
                                           database=database,
                                           from_time=from_time,
                                           to_time=to_time,
                                           low_confidence=low_confidence,
                                           low_snr=low_snr,
                                           delta=delta,
                                           residual_list=residual_list,
                                           batch_size=1)
    tp_list = [t[0][0] for t in tp]
    p_residual_list = [t[0][1][0] for t in tp]
    p_residual = []
    for i in p_residual_list:
        for j in i:
            p_residual.append(j)
    seisblue.plot.plot_error_distribution(p_residual)
    s_residual_list = [t[0][1][1] for t in tp]
    s_residual = []
    for i in s_residual_list:
        for j in i:
            s_residual.append(j)
    seisblue.plot.plot_error_distribution(s_residual)

    tp_list = np.array(tp_list).reshape(-1, 2)
    tp_list = tp_list.sum(axis=0)
    print(tp_list)
    return p_residual,s_residual


def compare_events(manual_events_list,assoc_events_list,plot=True):
    tp = 0
    tp_event = []
    tp_origin_event = []

    for origin_event in manual_events_list:
        from_time, to_time = seisblue.compare.get_ot_from_time_to_time(origin_event.time, 1)
        for assoc_events in assoc_events_list[:]:
            if to_time >= assoc_events.origin_time >= from_time:
                tp += 1
                assoc_events_list.remove(assoc_events)
                tp_event.append(assoc_events)
                tp_origin_event.append(origin_event)
                break
    print(f'true events number = {tp}')
    if plot:
        plot_event_residual(tp_event, tp_origin_event)