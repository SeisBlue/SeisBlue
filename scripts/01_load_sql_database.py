import seisblue

db = seisblue.sql.Client(database='demo', build=True)
inspector = seisblue.sql.DatabaseInspector(db)

db.read_hyp('/home/andy/Geom/STATION0.2020HP.HYP', 'HP')
# # db.read_kml_placemark('NCREE.kml','NCREE')
inspector.inventory_summery()
events = seisblue.io.read_event_list('/home/andy/Catalog/HP2020_04_05')
# # events = seisblue.io.read_afile_directory('/home/andy/Catalog/CWB2012/FEB')
db.add_sfile_events(events=events, tag="manual")

inspector.event_summery()
inspector.pick_summery()

inspector.plot_map(center=[121.5, 24], pad=1, depth_size=0.5, max_depth=160000)
