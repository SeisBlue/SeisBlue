# -*- coding: utf-8 -*-
import os
import subprocess
import glob
import re
import shutil
from tqdm import tqdm

from obspy.io.nordic.core import _write_nordic
from obspy.core.event.source import (FocalMechanism, NodalPlane)
from obspy.core.event import Magnitude, Origin

import seisblue
from seisblue import io, tool


def use_hypocenter(temp_filename, update=False):
    if update:
        hypo_result = subprocess.run(
            ['hyp', temp_filename, '-update'],
            input=b'y',
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    else:
        hypo_result = subprocess.run(
            ['hyp', temp_filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    return hypo_result


def use_fpfit(temp_filename):
    np = None
    fp_result = subprocess.run(
        ['fpfit', temp_filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    line = fp_result.stdout.split('\n')[-2]
    if not 'No solution' in str(line):
        np = NodalPlane(
            strike=eval(line[5:10]),
            dip=eval(line[16:20]),
            rake=eval(line[23:29]),
            strike_errors=eval(line[31:34]),
            dip_errors=eval(line[36:39]),
            rake_errors=eval(line[41:44]),
        )

    return np


def use_automag(temp_filename, debug=False):
    mags = []
    mag_result = subprocess.run(
        ['automag', temp_filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines = mag_result.stdout
    if debug:
        print(lines)
    for line in lines.split('\n'):
        if 'Average and median mw' in line:
            mags = re.findall(r'\d+\.\d+', line)
    return mags


def add_focal_mechanism(thres=10):
    sum_fm = 0
    events_path_list = glob.glob(f"./*L.S*")
    tool.check_dir('./fine', recreate=True)

    for event in tqdm(events_path_list):
        try:
            with open(event, 'r') as f:
                sfile_lines = f.readlines()
                ARC = [line.strip()[:-1].rstrip() for line in sfile_lines if
                       line.startswith(' ARC')]
            hyp = use_hypocenter(event)
            new_event_obspy = io.get_obspy_event('hyp.out')[0]

            np = use_fpfit(event)

            if np:
                high_quality = (np.strike_errors.uncertainty <= thres) and (
                        np.dip_errors.uncertainty <= thres) and (
                                       np.rake_errors.uncertainty <= thres)
                if high_quality:
                    fm = FocalMechanism(nodal_planes=np, method_id='FPFIT')
                    new_event_obspy.focal_mechanisms.append(fm)
                    _write_nordic(new_event_obspy, filename=event, wavefiles=ARC, outdir='./fine')
                    sum_fm += 1
        except Exception as e:
            print(e)

    print(f'Get {sum_fm} focal mechanism with error < {thres}.')


def add_magnitude(dataset_dir):
    sum_mag = 0
    events_path_list = glob.glob(f"./*L.S*")

    for event in tqdm(events_path_list):
        event_obspy = io.get_obspy_event(event)
        if event_obspy:
            event_obspy = event_obspy[0]
            hyp = use_hypocenter(event, update=True)
            mags = use_automag(event, debug=False)

            if mags:
                mags = [Magnitude(mag=mag, magnitude_type='Mw') for mag in mags]
                event_obspy.magnitudes = mags
                sum_mag += 1
                _write_nordic(event_obspy, filename=event, outdir='.')

    print(f'Add {sum_mag} magnitude to sfiles (outdir: {dataset_dir}).')


if __name__ == '__main__':
    config = seisblue.io.read_yaml('./config/data_config.yaml')
    c = config['global']
    result_dir = os.path.join('./result', c['dataset_name'])
    cwd = os.getcwd()
    os.chdir(result_dir)
    shutil.copy(c['hyp_filepath'], '.')
    shutil.copy(c['def_filepath'], '.')
    response_files = glob.glob(f'{c["response_dir"]}/*')
    [shutil.copy(resp, '.') for resp in response_files]

    # add_magnitude(result_dir)
    add_focal_mechanism(thres=c['threshold_error'])
    os.chdir(cwd)

