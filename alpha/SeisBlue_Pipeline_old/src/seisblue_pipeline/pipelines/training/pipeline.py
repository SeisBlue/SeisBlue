"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_dataset_paths,
                inputs="params:get_pretrain_dataset",
                outputs="dataset_paths",
            ),
            node(
                func=read_hdf5,
                inputs="dataset_paths",
                outputs=[
                    "instances_pretrain",
                    "dataset_pretrain"
                ],
            ),
            # node(
            #     func=plot_dataset,
            #     inputs="instances_pretrain",
            #     outputs=None,
            # ),
            node(
                func=get_dataloader,
                inputs=[
                    "params:get_pretrain_dataloader",
                    "dataset_pretrain",
                ],
                outputs=[
                    "train_loader",
                    "val_loader"
                ]
            ),
            node(
                func=train_test,
                inputs=[
                    "params:pretrain",
                    "train_loader",
                    "val_loader"
                ],
                outputs=None,
            ),
            node(
                func=evaluate,
                inputs=[
                    "params:pretrain",
                    "val_loader",
                    ],
                outputs=None,
            )
        ]
    )
