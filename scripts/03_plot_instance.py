import seisnn

db = seisnn.data.sql.Client('test.db')

picks = db.get_picks().all()
print(picks[0])

waveform = db.get_waveform(from_time=picks[0].time, to_time=picks[0].time,
                           station=picks[0].station).all()
print(waveform[0])

instance = seisnn.data.core.Instance(waveform[0])
instance.plot()
