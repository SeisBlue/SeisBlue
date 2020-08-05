import seisnn

db = seisnn.data.sql.Client('HL2017.db')

with db.session_scope() as session:
    Pick = db.get_table_class('pick')
    query = session.query(Pick)
    query = query.filter(Pick.phase.like('P'))

for row in query.all():
    print(row)
