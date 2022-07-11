# Custom SQL Query

seisblue SQL database is based on [sqlalchemy](https://www.sqlalchemy.org/) object relational mapper(orm),
check out [Querying](https://docs.sqlalchemy.org/en/13/orm/tutorial.html#querying) for further information. 

This script demonstrates how to write a custom query from SeisBlue.

```python
import datetime

import seisblue

# connect to sql database
db = seisblue.sql.Client('HL2017.db')

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
```

Output:

```text
Pick(Time=2017-01-23 13:02:07.430000, Station=H093, Phase=P, Tag=manual, SNR=None)
Pick(Time=2017-01-23 13:02:07.130000, Station=H098, Phase=P, Tag=manual, SNR=None)
Pick(Time=2017-01-23 13:02:07.500000, Station=H092, Phase=P, Tag=manual, SNR=None)
Pick(Time=2017-01-23 13:02:07.930000, Station=H084, Phase=P, Tag=manual, SNR=None)
```
