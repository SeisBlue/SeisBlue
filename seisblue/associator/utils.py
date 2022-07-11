import datetime
import os
import subprocess
import uuid

import numpy as np
import obspy
from obspy import UTCDateTime
from obspy.core.event import Pick, Event, Origin, WaveformStreamID, \
    EventDescription
from obspy.io.nordic.core import read_nordic, _write_nordic

import seisblue.utils


def call_hypocenter(match_candidates):
    config = seisblue.utils.Config()
    temp_filename = f'{str(uuid.uuid4())[0:7]}.tmp'
    if match_candidates:
        write_sfile(match_candidates, temp_filename)

    origin, picks = use_hypocenter(temp_filename)

    temp_file = os.path.join(config.associate, temp_filename)
    if os.path.exists(temp_file):
        os.remove(temp_file)

    return origin, picks

def use_hypocenter(temp_filename):
    config = seisblue.utils.Config()
    os.chdir(config.associate)
    hypo_result = subprocess.run(
        [os.path.join(config.seisan, 'PRO/hyp'), temp_filename],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    origin, picks = seisblue.io.read_hypout(hypo_result)
    return origin, picks

def hypo_search(match_candidates):
    QA = True

    origin, picks = call_hypocenter(match_candidates)

    arrivals = origin.arrivals
    pick_select = picks
    good_station = []
    if len(arrivals)%2==0:
        len_arr=len(arrivals)
    else:
        len_arr = len(arrivals)-1
    try:
        for i in range(0, len_arr, 2):
            p_residual = arrivals[i].time_residual
            s_residual = arrivals[i + 1].time_residual
            if abs(p_residual) <= 3 and abs(s_residual) <= 3:
                good_station.append(pick_select[i].waveform_id.station_code)
    except Exception as err:
        print(err)
        QA = False
        return match_candidates, origin, QA

    good_match_candidate = [candidate for candidate in match_candidates
                            if candidate.sta in good_station]
    if good_match_candidate:
        origin, picks = call_hypocenter(good_match_candidate)

    try:
        rms = origin.time_errors.uncertainty
        erdp = origin.depth_errors.uncertainty
        erln = origin.longitude_errors.uncertainty
        erlt = origin.latitude_errors.uncertainty
        used_sta_num = origin.quality.used_station_count
        az_gap = origin.quality.azimuthal_gap

    except Exception as err:
        print(err)
        QA = False
        return match_candidates, origin, QA
    try:
        if rms > 50 \
                or erdp > 50 \
                or erln > 50 \
                or erlt > 50:
            print(
                f'rms = {rms}, '
                f'erdp = {erdp}, '
                f'erln = {erln}, '
                f'erlt = {erlt}')
            QA = False
            return match_candidates, origin, QA
    except:
        print(
            f'rms = {rms}, '
            f'erdp = {erdp}, '
            f'erln = {erln}, '
            f'erlt = {erlt}')
        QA = False
        return match_candidates, origin, QA

    if used_sta_num <= 4 and az_gap >= 350:
        print(f'used_station_count = {used_sta_num} '
              f'azimuthal_gap = {az_gap}')
        QA = False
        return match_candidates, origin, QA

    match_candidates = [candidate for candidate in good_match_candidate
                        if candidate.sta in good_station]

    if len(match_candidates) < 3:
        QA = False

    return match_candidates, origin, QA


def write_sfile(match_candidates, temp_filename):
    config = seisblue.utils.Config()

    try:
        event_time = match_candidates[0].origin_time
    except Exception as err:
        print(err)
        return

    ev = Event()
    ev.event_descriptions.append(EventDescription())
    ev.origins.append(Origin(time=UTCDateTime(event_time),
                             latitude=0,
                             longitude=0,
                             depth=0))
    for candidate in match_candidates:
        _waveform_id_1 = WaveformStreamID(station_code=candidate.sta,
                                          channel_code='',
                                          network_code='')

        # obspy issue #2848 pick.second = 0. bug
        time = UTCDateTime(candidate.p_time)
        if time.second == 0 and time.microsecond == 0:
            time = time + 0.01
        ev.picks.append(Pick(waveform_id=_waveform_id_1,
                             phase_hint='P',
                             time=time,
                             evaluation_mode="automatic"))

        # obspy issue #2848 pick.second = 0. bug
        time = UTCDateTime(candidate.s_time)
        if time.second == 0 and time.microsecond == 0:
            time = time + 0.01
        ev.picks.append(Pick(waveform_id=_waveform_id_1,
                             phase_hint='S',
                             time=time,
                             evaluation_mode="automatic"))
    if ev.picks:
        _write_nordic(ev, filename=temp_filename, wavefiles=[],
                      outdir=config.associate)


def time_correction(datetime_list, norm='L2'):
    """
    mean, std = time_correction(datetime_list)

    Calculate the mean and standard deviations in seconds of
    a list of datetime values
    """
    offsets = []
    new_time = None
    for time_data in datetime_list:
        offsets.append((time_data - datetime_list[0]).total_seconds())

    if norm == 'L1':
        mean_offsets = np.mean(offsets)
        new_time = datetime_list[0] + \
                   datetime.timedelta(seconds=mean_offsets)
    elif norm == 'L2':
        median_offsets = np.median(offsets)
        new_time = datetime_list[0] + \
                   datetime.timedelta(seconds=median_offsets)
    elif norm == 'FA':
        tmp_time = datetime_list.copy()
        tmp_time.sort()
        new_time = tmp_time[0]

    std_offsets = np.std(offsets)
    return new_time, std_offsets


if __name__ == "__main__":
    pass
