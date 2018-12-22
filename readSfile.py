import matplotlib.pyplot as plt
import obspy.io.nordic.core as nordic
from obspy.core import *
import subprocess

waveFileDir = "/mnt/Data/HL/"
# lsOutput = subprocess.run(["ls", sfileDir], stdout=subprocess.PIPE, universal_newlines=True)
# sfileList = lsOutput.stdout.splitlines()

sfileList = []




for file in sfileList:
    catalog = nordic.read_nordic(file)
    catalog.wavename = nordic.readwavename(file)

    stream = Stream()
    for wave in catalog.wavename:
        stream += read(str(waveFileDir + wave))
    stream.sort(keys=['network', 'station', 'channel'])
    stream.normalize()
    start_time = catalog.events[0].origins[0].time
    stream.trim(start_time + 0, start_time + 1000)

    picked_stream = Stream()
    for event in catalog.events:
        for pick in event.picks:
            station = pick.waveform_id.station_code
            station_stream = stream.select(station=station)
            for trace in station_stream:
                trace.pick = pick
            picked_stream += station_stream

    for trace in picked_stream:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(trace.times("matplotlib"), trace.data, "k-")
        y_min, y_max = ax.get_ylim()
        pick_time = trace.pick.time.matplotlib_date
        pick_phase = trace.pick.phase_hint
        ax.vlines(pick_time, y_min, y_max, color='r', lw=2, label=pick_phase)
        plt.title(trace.id)
        ax.xaxis_date()
        fig.autofmt_xdate()
        ax.legend()
        plt.show()
