import datetime

import seisnn

# connect to sql database
db = seisnn.data.sql.Client('HL2017.db')

# open sql session, it will terminate the session when exit `with` statement
with db.session_scope() as session:

    # get Pick table class from name
    Pick = db.get_table_class('pick')

    # query all columns form Pick table
    query = session.query(Pick)

    # filter data with criteria
    query = query.filter(Pick.phase.like('P'))
    query = query.filter(Pick.time >= datetime.datetime(2017, 1, 23, 13, 2, 7))
    query = query.filter(Pick.time <= datetime.datetime(2017, 1, 23, 13, 2, 8))

# use .all() to make query into a list and print each row
for row in query.all():
    print(row)
