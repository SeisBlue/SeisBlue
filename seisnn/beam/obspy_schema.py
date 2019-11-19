import tensorflow as tf


def create_raw_metadata():
    raw_data_schema = {}

    raw_data_schema[KEY_COLUMN] = dataset_schema.ColumnSchema(
        tf.float32, [], dataset_schema.FixedColumnRepresentation())
