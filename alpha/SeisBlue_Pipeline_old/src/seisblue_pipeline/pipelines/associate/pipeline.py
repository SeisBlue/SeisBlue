"""
This is a boilerplate pipeline 'associate'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=create_associate_database,
            inputs=[
                "params:database",
                "params:associate"
            ],
            outputs=None,
            name="get_associate_picks",
        ),
    ])
