import numpy as np

def get_time_array(feature):
    time_array = np.arange(feature['npts'])
    time_array = time_array * feature['delta']
    return time_array


def select_phase(feature, phase):
    keys = list(feature['phase'].keys())
    for key in keys:
        if not phase in key:
            feature['phase'].pop(key)

    feature['picks'] = feature['picks'].loc[feature['picks']['pick_phase'] == phase]

    return feature


def select_channel(feature, channel):
    keys = list(feature['channel'].keys())
    for key in keys:
        if not channel in key:
            feature['channel'].pop(key)

    return feature


def select_pickset(feature, pickset):
    feature['picks'] = feature['picks'].loc[feature['picks']['pick_set'] in pickset]

    return feature
