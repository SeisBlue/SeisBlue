"""
This is a boilerplate test file for pipeline 'data_processing'
generated using Kedro 0.18.2.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
from kedro.extras.datasets.text import TextDataSet
from kedro.extras.datasets.pickle import PickleDataSet
from kedro.extras.datasets.pandas import CSVDataSet
from pandas.testing import assert_frame_equal

from src.seisblue_pipeline.pipelines.data_processing.nodes import read_hyp, \
    add_network, make_event_list, read_event_list, make_dataset_dataframe


def test_read_hyp_and_return_geom_dict():
    inputs = TextDataSet(filepath="./data/STATION0.HYP.txt").load()
    outputs = read_hyp(inputs)
    target = PickleDataSet(filepath='./data/geom_dict.pkl').load()
    assert outputs == target


def test_read_geom_dict_and_add_network():
    inputs = PickleDataSet(filepath='./data/geom_dict.pkl').load()
    outputs = add_network(inputs, 'HP')
    target = PickleDataSet(filepath='./data/geom_network_dict.pkl').load()
    assert outputs == target


def test_make_event_list():
    outputs = make_event_list('/mnt/sfile/HP2017')
    target = PickleDataSet(filepath='./data/event_list.pkl').load()
    assert outputs == target


def test_read_event_list():
    inputs = PickleDataSet(filepath='./data/event_list.pkl').load()
    outputs = read_event_list(inputs)
    target = PickleDataSet(filepath='./data/events.pkl').load()
    assert len(outputs) == len(target)
    assert outputs[0]['event_type'] == target[0]['event_type'] == 'earthquake'


def test_make_dataset_dataframe():
    inputs_geom = PickleDataSet(filepath='./data/geom_network_dict.pkl').load()
    inputs_event = PickleDataSet(filepath='./data/events.pkl').load()
    outputs = make_dataset_dataframe(inputs_geom, inputs_event, 'manual')
    target_geom = CSVDataSet(filepath='./data/preprocessed_geom.csv').load()
    target_event = CSVDataSet(filepath='./data/preprocessed_event.csv').load()
    target_pick = CSVDataSet(filepath='./data/preprocessed_pick.csv').load()

    assert outputs[0]['Location'].astype(str).all() \
           == target_geom['Location'].all()
    assert outputs[1]['Latitude'].all() == target_event['Latitude'].all()
    assert outputs[2]['Station'].all() == target_pick['Station'].all()


