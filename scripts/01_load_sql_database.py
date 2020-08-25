import seisnn.data

db = seisnn.data.sql.Client("HL2017.db")

db.read_hyp("HL2017.HYP", network="HL2017")
db.inventory_summery()

db.add_events("HL2017", tag="manual")
db.event_summery()
db.pick_summery()

db.plot_map()
