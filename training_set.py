import obspy.io.nordic.core as nordic
from obspy.core import Stream, read
from obspy.core.event.catalog import Catalog


def load_catalog(sfileList):
    catalog = Catalog()
    for file in sfileList:
        event = nordic.read_nordic(file)
        event.wavename = nordic.readwavename(file)
        catalog += event
    return catalog


def get_waveform(waveFileDir, start_time, duration):
    stream = Stream()
    for wave in event.wavename:
        stream += read(str(waveFileDir + wave))
    stream.normalize()
    start_time = event.events[0].origins[0].time
    stream.trim(start_time + 0, start_time + duration)

    return waveform


def get_probability(catalog, waveform):
    return probability


def split_dataset(waveform, p_prob):
    return (train_waveform, train_p_prob), (test_waveform, test_p_prob)


def load_data(sfileList, waveFileDir):
    catalog = load_catalog(sfileList)
    waveform = get_waveform(catalog, waveFileDir)
    p_prob = get_probability(catalog, waveform)
    (train_waveform, train_p_prob), (test_waveform, test_p_prob) = split_dataset(waveform, p_prob)

    return (train_waveform, train_p_prob), (test_waveform, test_p_prob)
