"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline
from .pipelines import data_processing as dp
from .pipelines import generate_dataset as gd
from .pipelines import training as tr
from .pipelines import associate as assoc


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    data_processing_pipeline = dp.create_pipeline()
    generate_dataset_pipeline = gd.create_pipeline()
    training_pipeline = tr.create_pipeline()
    associate_pipeline = assoc.create_pipeline()

    return {"__default__": data_processing_pipeline + generate_dataset_pipeline,
            "dp": data_processing_pipeline,
            "gd": generate_dataset_pipeline,
            "tr": training_pipeline,
            "assoc": associate_pipeline,
            }

