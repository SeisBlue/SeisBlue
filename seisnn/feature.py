import numpy as np


def pick_exist(feature):
    if not feature['pick_phase'][0] == 'NA':
        return True
    else:
        return False


def get_time_array(feature):
    time_array = np.arange(feature['npts'])
    time_array = time_array * feature['delta']
    return time_array