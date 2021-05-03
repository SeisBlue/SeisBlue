import seisnn

db = seisnn.sql.Client(database="tt.db")
inspector = seisnn.sql.DatabaseInspector(db)

# db.read_nsta(nsta="nsta24.dat")
# inspector.inventory_summery()

db.read_hyp(hyp="HL2019.HYP", network="HL")
inspector.inventory_summery()


# path_list = '/home/andy/A_file/*'
# events = seisnn.io.read_afile_directory(path_list)

db.add_events(catalog="HL2019", tag="manual")
inspector.event_summery()
inspector.pick_summery()

inspector.plot_map()
