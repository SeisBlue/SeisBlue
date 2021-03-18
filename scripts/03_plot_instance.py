import seisnn

database = 'Hualien.db'
db = seisnn.sql.Client(database=database)

waveforms = db.get_waveform()
for waveform in waveforms:
    instance = seisnn.core.Instance(waveform)
    instance.plot()
