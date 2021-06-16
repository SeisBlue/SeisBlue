import seisnn
from datetime import datetime
from obspy import UTCDateTime
import numpy as np


def get_from_time_to_time(pick, delta=0.1):
    from_time = UTCDateTime(pick.time) - delta
    from_time = datetime.strptime(str(from_time), '%Y-%m-%dT%H:%M:%S.%fZ')
    to_time = UTCDateTime(pick.time) + delta
    to_time = datetime.strptime(str(to_time), '%Y-%m-%dT%H:%M:%S.%fZ')
    return from_time, to_time


def compare(station, database, delta=0.2):
    tp = [0, 0]
    db = seisnn.sql.Client(database=database)
    P_picks = db.get_picks(tag='manual', phase='P', station=station.station)
    S_picks = db.get_picks(tag='manual', phase='S', station=station.station)

    for i, picks in enumerate([P_picks, S_picks]):
        for pick in picks:

            from_time, to_time = get_from_time_to_time(pick, delta)
            correspond_pick = db.get_picks(from_time=from_time, to_time=to_time, tag='predict', phase=pick.phase,
                                           station=pick.station)
            if correspond_pick:
                tp[i] = tp[i] + 1
        print(station)
        print(f'total in station: {len(picks)} true positive {tp[i]}')
    return tp

def main():
    stations = seisnn.sql.Client(database="Hualien.db").get_inventories()
    tp = seisnn.utils.parallel(stations,
                          func=compare,
                          database="Hualien.db",
                          delta=0.1,
                          batch_size=1)
    tp = np.array(tp).reshape(-1,2)
    tp = tp.sum(axis = 0)
    print(tp)
if __name__ == '__main__':
        main()


