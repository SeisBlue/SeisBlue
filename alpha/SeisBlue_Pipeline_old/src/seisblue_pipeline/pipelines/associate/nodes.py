"""
This is a boilerplate pipeline 'associate'
generated using Kedro 0.18.2
"""

from ...seisblue import core
from ...seisblue.associator import core as assoc_core


def create_associate_database(database, params):
    db = core.Client(database=database)
    P_picks = db.get_picks(
        tag=params['tag'],
        station=params['station'],
        from_time=params['from_time'],
        to_time=params['to_time'],
        phase="P",
        # low_snr=0.3
        # low_confidence=0.3
        # high_confidence=2.5,
    )
    S_picks = db.get_picks(
        tag=params['tag'],
        station=params['station'],
        from_time=params['from_time'],
        to_time=params['to_time'],
        phase="S",
    )
    picks = [pick for pick in P_picks] + [pick for pick in S_picks]

    db_assoc = assoc_core.Client(database=params['assoc_database'], build=True)
    db_assoc.read_picks(picks, params['waveforms_dir'])
