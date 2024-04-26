"""
This is a boilerplate pipeline 'generate_dataset'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_picks,
                inputs=[
                    "params:database",
                    "params:tag"
                ],
                outputs="picks_use",
            ),
            node(
                func=get_waveform_time_windows,
                inputs=[
                    "picks_use",
                    "params:trace_length"
                ],
                outputs="time_windows"
            ),
            node(
                func=get_waveforms,
                inputs=[
                    "time_windows",
                    "params:waveforms_dir"
                ],
                outputs="streams",
            ),
            node(
                func=signal_preprocessing,
                inputs="streams",
                outputs="processed_streams",
            ),
            node(
                func=get_instances,
                inputs=[
                    "processed_streams",
                    "params:database",
                    "params:phase",
                    "params:tag",
                    "params:shape",
                    "params:half_width",
                ],
                outputs="instances",
            ),
            node(
                func=write_hdf5,
                inputs=[
                    "instances",
                    "params:HDF5_filepath",
                    ],
                outputs=None,
            ),
            node(
                func=add_waveforms,
                inputs=[
                    "instances",
                    "params:database",
                    "params:HDF5_filepath",
                ],
                outputs=None,
            ),
        ]
    )
