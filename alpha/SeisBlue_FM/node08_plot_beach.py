# -*- coding: utf-8 -*-
from seisblue import io, tool

from tqdm import tqdm
import os
import subprocess


def run_gmt(filename, debug=False):
    result = subprocess.run(
        ['bash', filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    if debug:
        print(result)


def prepare_meca_file(event, output_filename):
    lines = []
    with open(output_filename, 'w') as f:
        longitude = event.origins[0].longitude
        latitude = event.origins[0].latitude
        depth = event.origins[0].depth / 1000  # km
        np1 = event.focal_mechanisms[0].nodal_planes.nodal_plane_1
        strike = np1.strike
        dip = np1.dip
        rake = np1.rake
        magnitude = event.magnitudes[0].mag if len(event.magnitudes) > 0 else 1
        origin_time = event.origins[0].time
        line = '{} {} {} {} {} {} {} {} {} {}\n'.format(longitude, latitude, depth, strike, dip, rake, magnitude, 0, 0, origin_time)
        f.write(line)
        lines.append(line)
    return lines


def prepare_polarity_file(event, output_filename):
    stations = []
    with open(output_filename, 'w') as f:
        for pick in event.picks:
            station = pick.waveform_id.station_code
            arrival = [arr for arr in event.origins[0].arrivals if
                       arr.pick_id == pick.resource_id][0]
            azimuth = arrival.azimuth
            takeoff_angle = int(arrival.takeoff_angle)
            polarity = pick.polarity
            if polarity == 'positive':
                polarity = 'c'
            elif polarity == 'negative':
                polarity = 'd'
            else:
                polarity = 'x'
            if station not in stations and polarity != 'x':
                stations.append(station)
                f.write('{} {} {} {}\n'.format(station, azimuth, takeoff_angle, polarity))


def prepare_meca_all(lines, output_filename):
    with open(output_filename, 'w') as f:
        f.write('\n'.join(lines))


def plot_beach(events, gmt_dir):
    cwd = os.getcwd()
    os.chdir(gmt_dir)
    for event in tqdm(events):
        lines = prepare_meca_file(event, 'meca.txt')
        prepare_polarity_file(event, 'polarity.txt')
        prepare_meca_all(lines, 'meca_all.txt')
        run_gmt('beach_polarity.sh')
        run_gmt('beach.sh')
    os.chdir(cwd)


if __name__ == '__main__':
    config = io.read_yaml('./config/data_config.yaml')
    c = config['plot_beach']

    events_dir = os.path.join(f"./result/{c['dataset_name']}/fine")
    events = io.get_obspy_events(events_dir)

    plot_beach(events, './gmt_script')
