import seisblue

config = seisblue.utils.Config()
database = 'assoc_HP2020_GAN_N_noise_04_05_snr_03'
associates = seisblue.sql.get_associates(database=database,
                                         from_time='2020-04-30 19:20:00',
                                         to_time='2020-04-30 19:22:00')

seisblue.utils.parallel(associates,
                        func=seisblue.io.associate_to_sfile,
                        database=database,
                        out_dir='/home/andy/Catalog/associate_sfile_output',
                        batch_size=16)
