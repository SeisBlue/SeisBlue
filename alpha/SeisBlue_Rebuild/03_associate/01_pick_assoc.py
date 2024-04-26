# -*- coding: utf-8 -*-
import argparse
from obspy.clients.filesystem import sds
import pandas as pd
from tqdm import tqdm
import shutil

from seisblue import SQL, plot, utils, io, core


def get_picks(database, **kwargs):
    client = SQL.Client(database)
    results = client.get_picks(**kwargs)
    picks = [pick.to_dataclass() for pick in results]
    return picks


def get_nslc(sds_root):
    client = sds.Client(sds_root=sds_root)
    nslc = client.get_all_nslc()
    df_nslc = pd.DataFrame(nslc, columns=["net", "sta", "loc", "chan"])
    df_nslc = df_nslc.drop(columns="chan").drop_duplicates()
    df_nslc = df_nslc.set_index("sta")
    return df_nslc


def _pickassoc(pick, df_nslc):
    try:
        if isinstance(df_nslc.loc[pick.inventory.station]["net"], pd.Series):
            df_nslc_net = df_nslc.loc[pick.inventory.station]["net"][0]
        else:
            df_nslc_net = df_nslc.loc[pick.inventory.station]["net"]
    except:
        df_nslc_net = None
    try:
        if isinstance(df_nslc.loc[pick.inventory.station]["loc"], pd.Series):
            df_nslc_loc = df_nslc.loc[pick.inventory.station]["loc"][0]
        else:
            df_nslc_loc = df_nslc.loc[pick.inventory.station]["loc"]
    except:
        df_nslc_loc = None

    item = core.PickAssoc(
        sta=pick.inventory.station,
        net=df_nslc_net,
        loc=df_nslc_loc,
        time=pick.time,
        snr=pick.snr,
        trace_id=pick.traceid,
        phase=pick.phase,
    )
    item = core.PickAssocSQL(item)
    return item


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config_filepath", type=str, required=True)
    args = parser.parse_args()
    config = io.read_yaml(args.data_config_filepath)
    c = config['associator']
    database = c['database']

    shutil.copy(c['hyp_filepath'], '.')
    client = SQL.Client(database)

    # client.get_candidate(assoc_id=1)
    # for i in range(3, 10):
    client.analysis_candidate(assoc_id=1)









