import seisblue

txt_path = '/home/andy/event.txt'

associates1 = seisblue.sql.get_associates(database="SF19.db")
associates2 = seisblue.sql.get_associates(database="SF20.db")
associates3 = seisblue.sql.get_associates(database="SF21.db")
associates4 = seisblue.sql.get_associates(database="SF22.db")
associates5 = seisblue.sql.get_associates(database="SF23.db")
associates6 = seisblue.sql.get_associates(database="SF24.db")
associates7 = seisblue.sql.get_associates(database="SF25_02.db")
associates8 = seisblue.sql.get_associates(database="SF03_13.db")
associate_list = [associates1, associates2, associates3, associates4,
                  associates5, associates6, associates7, associates8]

seisblue.io.associate_to_txt(txt_path, associate_list)
