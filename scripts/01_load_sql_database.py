import seisnn.data

db = seisnn.data.sql.Client("HL2019.db")

db.read_hyp("HL2019.HYP", network="HL2019")
db.inventory_summery()

db.add_events("HL2019", tag="manual")
db.event_summery()
db.pick_summery()

db.plot_map()
