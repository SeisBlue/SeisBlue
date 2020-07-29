from seisnn import sql

db = sql.Client("test.db")

db.read_hyp("HL2017.HYP", network="HL2017")
db.geom_summery()

db.add_events("test", tag="manual")
db.event_summery()
db.pick_summery()

db.plot_map()
