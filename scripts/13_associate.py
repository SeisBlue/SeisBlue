import time

import seisblue

db = seisblue.sql.Client(database='demo')
P_picks = db.get_picks(tag='manual',
                       station='*',
                       from_time='2012-01-01 00:00:00',
                       to_time='2023-01-31 23:59:59',
                       phase='P',
                       # low_snr=0.3
                       # low_confidence=0.3
                       # high_confidence=2.5,
                       )
S_picks = db.get_picks(tag='manual',
                       station='*',
                       from_time='2012-01-01 00:00:00',
                       to_time='2023-01-31 23:59:59',
                       phase='S',
                       # low_confidence=0.3
                       )
# P_picks,S_picks = seisblue.compare.compare_dataset_pick('CWB_10_EH_HH_HL_trace', from_time='2012-01-01 00:00:00',
#                          to_time='2012-02-01',low_confidence=0.5,residual_list=True)
picks = []
for pick in P_picks:
    picks.append(pick)
for pick in S_picks:
    picks.append(pick)

db_assoc = seisblue.associator.sql.Client('demo_assoc',
                                          build=True)
db_assoc.read_picks(picks)

associator = seisblue.associator.core.LocalAssociator(
    'demo_assoc',
    max_s_p=60,
    origin_time_delta=3,
    nsta_declare=3)

# candidate events
print('candidate events')
associator.id_candidate_events()
print('associate events')
associator.fast_associate(plot=True)
