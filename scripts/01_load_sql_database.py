import seisnn

db = seisnn.sql.Client(database="Hualien.db")

db.read_hyp(hyp="HL2019.HYP", network="HL2019")
db.inventory_summery()

db.add_events(catalog="HL2019", tag="manual")
db.event_summery()
db.pick_summery()

db.plot_map()
