import seisblue

database = 'demo'
db = seisblue.sql.Client(database=database)

waveforms = db.get_waveform()
for waveform in waveforms:
    instance = seisblue.core.Instance(waveform)
    instance.plot()
