import seisblue

db = seisblue.sql.Client(database='HC2021.db')
picks = db.get_picks(tag='predict',phase='P')
for pick in picks:
    tfr_converter = seisblue.components.TFRecordConverter()
    metadata = tfr_converter.get_time_window(anchor_time=pick.time,
                                    station=pick.station,
                                    shift=10)

    streams = seisblue.io.read_sds(metadata)
    for _, stream in streams.items():
        stream.detrend('linear')
        stream.detrend('demean')
        stream.filter('bandpass', freqmin=1, freqmax=45)
        stream.plot()
