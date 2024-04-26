"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_stations_time_window,
                inputs="params:waveforms_dir",
                outputs="stations_time_window",
            ),
            node(
                func=read_hyp,
                inputs=[
                    "hyp_file",
                    "stations_time_window",
                ],
                outputs="inventories",
            ),
            node(
                func=read_sfile,
                inputs="params:events_dir",
                outputs="obspy_events",
            ),
            node(
                func=get_events_from_obspy_events,
                inputs="obspy_events",
                outputs="events",
            ),
            node(
                func=get_picks_from_obspy_events,
                inputs=[
                    "obspy_events",
                    "params:tag"
                ],
                outputs="picks",
            ),
            # node(
            #     func=create_database,
            #     inputs=[
            #         "params:database",
            #         "inventories",
            #         "events",
            #         "picks"
            #     ],
            #     outputs=None,
            # ),
            # node(
            #     func=check_pick,
            #     inputs="params:database",
            #     outputs=None,
            # ),
        ]
    )
