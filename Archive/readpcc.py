import matplotlib.pyplot as plt
import numpy as np
from obspy.core import *


class Network(Stream):
    def __init__(self):
        super().__init__()
        self.data_block = None

    def fill_missing_station(self, stations, trace_num):
        self._get_data_block()
        data_points = len(self.data_block[0])
        missing_traces = [0]
        for i in range(0, len(stations)):
            start_trace = (stations[i] - 1) * trace_num + 1
            end_trace = (stations[i]) * trace_num
            trace_list = range(start_trace, end_trace + 1)
            missing_traces.append(trace_list)

        for i in missing_traces:
            self.data_block = np.insert(self.data_block, i, np.zeros(data_points), axis=0)

        return self.data_block

    def top_mute(self, time_length):
        sample_rate = int(round(self.traces[0].stats.sampleing_rate))
        mute_data_length = int(time_length * sample_rate)

        self.data_block[:, 0:mute_data_length] = 0
        return self.data_block

    def plot_section(self, station_list):
        section = self._select_traces(station_list)
        data_points = len(self.data_block[0])
        station_number = len(station_list)

        vm = np.percentile(self.data_block, 97)
        plt.figure(figsize=(station_number, 10))
        plt.xticks(range(station_number), station_list)
        plt.yticks(range(0, data_points, 25), range(0, 11))
        plt.imshow(section.T, cmap="Greys", vmin=-vm, vmax=vm, aspect='auto')

        plt.colorbar()
        plt.show()

    def _select_traces(self, station_list):
        section = np.zeros(len(self.data_block[0]))
        for station in station_list:
            selected_trace = self.data_block[station]
            section = np.vstack((section, selected_trace))

        section = np.delete(section, 0, 0)

        return section

    def _get_data_block(self):
        self.data_block = np.stack(t.data for t in self.traces)


def read_file(filelist):
    with open(filelist, "r") as file:
        lines = file.read().splitlines()
    lines.sort()

    st = Network()
    for l in lines:
        st += read(l)
    return st


station_plot_list = [[12, 17, 23, 28, 34, 39, 46, 52],
                     [6, 11, 16, 22, 27, 33, 45, 51, 58],
                     [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]]

missing_stations = [25, 29]

st = read_file("filelist.txt")

tr = st[0]
start_time = tr.stats.starttime
st.trim(start_time, start_time + 10)
st.filter('bandpass', freqmin=0.7, freqmax=1.5, corners=2, zerophase=True)
st.normalize(global_max=True)
st.fill_missing_station(stations=missing_stations, trace_num=1)
st.top_mute(time_length=0.25)

for row in station_plot_list:
    st.plot_section(row)
