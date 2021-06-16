import seisnn

db = seisnn.sql.Client(database="CWB.db")
inspector = seisnn.sql.DatabaseInspector(db)

db.read_GDMSstations(nsta="GDMSstations.csv")
inspector.inventory_summery()
# db.read_hyp(hyp="HL2019.HYP", network="HL")
# inspector.inventory_summery()


path_list = '/home/andy/A_file/*'
events = seisnn.io.read_afile_directory(path_list)
# events = seisnn.io.read_event_list('HL2018')
db.add_sfile_events(events=events, tag="manual")

inspector.event_summery()
inspector.pick_summery()

inspector.plot_map()
