import seisblue

station_database = 'HP2020'
db = seisblue.sql.Client(station_database)
events = seisblue.io.read_event_list('associate_sfile_output')
for event in events:
    seisblue.plot.plot_event_trace(db, event, 'HP')
