# -*- coding: utf-8 -*-
from strec.strec import kagan
import os
import matplotlib.pyplot as plt
import glob

from obspy.imaging.beachball import beach, beachball

from seisblue import core, io


def get_nodel_planes(obspy_event):
    nodel_planes = {}
    for fm in obspy_event.focal_mechanisms:
        np = fm.nodal_planes.nodal_plane_1
        method = str(fm.method_id).split('/')[-1]
        if np.strike_errors.uncertainty:
            np = core.NodalPlane(strike=np.strike,
                                 strike_errors=np.strike_errors.uncertainty,
                                 dip=np.dip,
                                 dip_errors=np.dip_errors.uncertainty,
                                 rake=np.rake,
                                 rake_errors=np.rake_errors.uncertainty,
                                 method=method)
        else:
            np = core.NodalPlane(strike=np.strike,
                                 dip=np.dip,
                                 rake=np.rake,
                                 method=method)
        nodel_planes[method] = np
    return nodel_planes


def calculate_kagan(obspy_events, method_id):
    kagan_events = []
    for obspy_event in obspy_events:
        if hasattr(obspy_event, 'focal_mechanisms') and len(
                obspy_event.focal_mechanisms) > 1:
            nodel_planes = get_nodel_planes(obspy_event)
            np1 = nodel_planes['ori']
            np2 = nodel_planes[method_id]

            kagan_angle = kagan.get_kagan_angle(np1.strike,
                                                np1.dip,
                                                np1.rake,
                                                np2.strike,
                                                np2.dip,
                                                np2.rake)
            mag = obspy_event.magnitudes[0].mag if len(
                obspy_event.magnitudes) > 0 else None

            if np2.strike_errors:
                high_quality = (np2.strike_errors < 15) and (
                        np2.dip_errors < 15) and (
                                       np2.rake_errors < 15)
            else:
                high_quality = True

            if high_quality:
                k = core.Kagan(np1=np1, np2=np2, kagan_angle=kagan_angle)
                event = core.Event(time=obspy_event.origins[0].time,
                                   magnitude=mag,
                                   kagan=k)
                kagan_events.append(event)

    return kagan_events


def plot_kagan_histogram(kagan_events, title='', fig_dir='.',
                         filepath='distribution_of_kagan_angles.png',
                         plot=True):
    kagan_angles = [kagan_event.kagan.kagan_angle for kagan_event in
                    kagan_events]
    small_count = len(
        [kagan for kagan_angle in kagan_angles if kagan_angle < 40])

    big_kagan = '\n'.join(
        [str(kagan_event.time) + ' ' + str(kagan_event.magnitude) for
         kagan_event in kagan_events if kagan_event.kagan.kagan_angle > 40])
    with open('bad_kagan_angles.txt', 'w') as f:
        f.write(big_kagan)
    small_percentage = round(small_count / len(kagan_angles) * 100) if len(
        kagan_angles) > 0 else 0
    if plot and small_percentage:
        print("Plot distribution of kagan angles.")
        plt.figure()
        plt.hist(kagan_angles, 10)
        plt.axvline(40, linestyle='--', color='gray')

        plt.xlabel("Angle")
        plt.ylabel("Count")

        plt.title(
            f'{title}\n(Kagan angle < 40)/Total: {small_count}/{len(kagan_angles)} ({small_percentage}%)')
        plt.savefig(os.path.join(fig_dir, filepath),
                    dpi=300)
    return (small_count, len(kagan_angles), small_percentage)


def plot_focal_mechanism(kagan_events, threshold, title='',
                         fig_path='kagan_example_big.png'):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.plot([0, 160], [0, 40], "rv", ms=0.1)
    ax = plt.gca()

    sub_kagan_events = [kagan_event for kagan_event in
                        sorted(kagan_events, key=lambda x: x.kagan.kagan_angle)
                        if kagan_event.kagan.kagan_angle > threshold]

    for i in range(10):
        try:
            kg = sub_kagan_events[i].kagan
            np1 = (kg.np1.strike, kg.np1.dip, kg.np1.rake)
            np2 = (kg.np2.strike, kg.np2.dip, kg.np2.rake)

            if kg.np1.strike_errors:
                np1_errors = tuple(map(int, (
                    kg.np1.strike_errors, kg.np1.dip_errors,
                    kg.np1.rake_errors)))
            else:
                np1_errors = ''
            if kg.np2.strike_errors:
                np2_errors = tuple(map(int, (
                    kg.np2.strike_errors, kg.np2.dip_errors,
                    kg.np2.rake_errors)))
            else:
                np2_errors = ''
            y = (i // 10 + 1) * 30
            x = (i % 10 + 1) * 15
            beach1 = beach(np1, xy=(x, y), width=8, linewidth=0.5)
            beach2 = beach(np2, xy=(x, y - 10), width=8, linewidth=0.5)
            ax.add_collection(beach1)
            ax.add_collection(beach2)
            ax.text(x - 3, 5, f'{int(kg.kagan_angle)}')
            ax.text(x - 3, 0, f'{np1_errors}', fontsize=5)
            ax.text(x - 3, -5, f'{np2_errors}', fontsize=5)
            ax.text(x - 3, -10, f'{sub_kagan_events[i].magnitude}', fontsize=5)
        except IndexError as e:
            print(f'The number of results are less than 10.')

    ax.text(-35, 30, 'predict')
    ax.text(-35, 20, 'manual')
    ax.text(-45, 5, 'kagan angle')
    ax.text(-35, -5, 'error')
    ax.text(-35, -12, 'Mw')
    ax.set_aspect("equal")
    # ax.set_xlim((-50, 160))
    # ax.set_ylim((-10, 40))
    plt.axis('off')
    plt.savefig(fig_path, dpi=300)
    print('Plot Kagan results and beach balls.')


def kagan_test(filepath, plot=True):
    config = io.read_yaml('./config/data_config.yaml')
    c = config['kagan']

    result_dir = glob.glob(f'./result/{c["dataset_name"]}/fine/best')[0]
    obspy_events = io.get_obspy_events(result_dir)
    kagan_events = calculate_kagan(obspy_events, c['method_id'])
    print(f'Get {len(kagan_events)} events.')
    (small_count, all_count, small_percentage) = plot_kagan_histogram(kagan_events,
                                            title=f'Distribution of kagan_angles for {c["dataset_name"]}',
                                            fig_dir=f'./figure/{c["dataset_name"]}',
                                            filepath=filepath,
                                            plot=plot)

    # plot_focal_mechanism(kagan_events,
    #                      threshold=35,
    #                      title=f'kagan example for {c["dataset_name"]}',
    #                      fig_path=f'./figure/{c["dataset_name"]}/kagan_example')
    return (small_count, all_count, small_percentage)



