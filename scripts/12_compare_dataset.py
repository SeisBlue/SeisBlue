import seisblue
from seisblue.compare import compare_events

seisblue.compare.compare_dataset_pick('HP2020',
                                      from_time='2020-01-01 00:00:00',
                                      to_time='2020-06-01', station='*',
                                      low_confidence=0.1, delta=0.5,
                                      residual_list=True)

db = seisblue.sql.Client('HP2020_N_manual')
events = db.get_events()
assoc_events_list = seisblue.sql.get_associates(
    'assoc_HP2020_GAN_N_noise_04_05', max_erlim=50)
compare_events(events, assoc_events_list)
