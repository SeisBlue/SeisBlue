import seisnn

db = seisnn.data.sql.Client('HL2019.db')

picks = db.get_picks(phase='S').all()

for pick in picks:
    waveform = db.get_waveform(from_time=pick.time, to_time=pick.time,
                               station=pick.station).all()
    instance = seisnn.data.core.Instance(waveform[0])
    instance.plot()
