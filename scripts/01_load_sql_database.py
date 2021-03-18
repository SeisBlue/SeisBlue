import seisnn

db = seisnn.sql.Client(database="Hualien.db")
inspector = seisnn.sql.DatabaseInspector(db)

db.read_hyp(hyp="HL2019.HYP", network="HL2019")
inspector.inventory_summery()

db.add_events(catalog="HL2019", tag="manual")
inspector.event_summery()
inspector.pick_summery()

inspector.plot_map()
