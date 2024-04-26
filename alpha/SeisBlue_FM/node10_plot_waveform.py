# -*- coding: utf-8 -*-
import os
import glob
import matplotlib.pyplot as plt
import waveform_processing as wf
from itertools import chain
import matplotlib.dates as mdates
from functools import partial
import shutil
from obspy import UTCDateTime

from bokeh.plotting import figure, show
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, LinearColorMapper, HoverTool, \
    BasicTicker, ColorBar, PrintfTickFormatter, Slider, CustomJS, CheckboxGroup, \
    Circle, Range1d, BoxAnnotation, Select, WMTSTileSource, Spacer, Div, TapTool, DateRangeSlider
from bokeh.layouts import gridplot, column


from seisblue import io, tool


def plot_waveforms(dataset_name, trace_length, waveforms_dir, fig_dir, event_id):
    file = glob.glob(f'./dataset/{dataset_name}_{event_id[:7]}_test.hdf5')[0]
    event_instance = io.read_hdf5_event(file, key=event_id)[0]

    pick = event_instance.instances[0].Ppick
    time_window = wf.get_waveform_time_windows(pick,
                                               trace_length=trace_length)
    P_picks = [instance.Ppick for instance in event_instance.instances]
    S_picks = [instance.Spick for instance in event_instance.instances]

    stations = [instance.Ppick.inventory.station for instance in
                event_instance.instances]
    stream = wf.get_waveforms(time_window, stations, waveforms_dir)
    print(f'Get {len(stream)} stream.')

    processed_waveforms = tool.parallel(stream,
                                        func=wf.signal_preprocessing)
    processed_waveforms = list(chain.from_iterable(
        sublist for sublist in processed_waveforms if sublist))
    print(f'Get {len(processed_waveforms)} processed waveforms.')

    for i, stream in enumerate(processed_waveforms):
        station = stream[0].stats.station
        fig = plt.figure()
        stream.plot(fig=fig)

        for ax in fig.axes:
            Ppick_time_matplotlib = mdates.date2num(
                UTCDateTime(P_picks[i].time).datetime)
            Spick_time_matplotlib = mdates.date2num(
                UTCDateTime(S_picks[i].time).datetime)
            ax.axvline(Ppick_time_matplotlib, color='blue',
                       linestyle='--',
                       linewidth=1, label='predict P')
            ax.axvline(Spick_time_matplotlib, color='red',
                       linestyle='--',
                       linewidth=1, label='predict S')
        fig.axes[0].legend()
        fig.axes[0].set_title(event_instance.event.time)
        filepath = os.path.join(fig_dir, f'{station}.jpg')
        plt.savefig(filepath)
        plt.close()


def plot_waveform_wab(fig_dir):
    event_id = fig_dir.split('/')[-1]
    output_file(f'{event_id}.html')
    img_urls = glob.glob(f'{fig_dir}/*')
    img_urls.sort()
    stations = [url.split('/')[-1].split('.')[0] for url in img_urls]

    div = Div(width=500, height=500)
    select = Select(title="Select Station:", value="0",
                    options=[(str(i), f"{stations[i]}") for i in range(len(img_urls))])
    slider = Slider(start=0, end=len(img_urls)-1, value=0, step=1, title="Image Index")

    callback = CustomJS(args=dict(div=div, img_urls=img_urls, select=select, slider=slider), code="""
        const index = cb_obj.value;
        div.text = '<img src="' + img_urls[index] + '" alt="image">';
        select.value = index.toString();
        slider.value = parseInt(index);
    """)
    select.js_on_change('value', callback)
    slider.js_on_change('value', callback)

    # 初始化Div的顯示內容
    div.text = f'<img src="{img_urls[0]}" alt="image">'

    # 布局
    controls_column = column(select, slider)
    layout = gridplot([[Spacer(width=350), div, Spacer(width=300), controls_column]])

    show(layout)


def cleanup(fig_dir):
    event_id = fig_dir.split('/')[-1]
    try:
        os.remove(f'{event_id}.html')
        shutil.rmtree(fig_dir)
    except Exception as e:
        print(f"Error during cleanup: {e}")


if __name__ == '__main__':
    config = io.read_yaml('./config/data_config.yaml')
    c = config['plot_map']
    event_id = str(UTCDateTime(c['event_id']))[:-1]
    print(event_id)

    fig_dir = os.path.join(f"./figure/{c['dataset_name']}/waveform/{event_id}")
    tool.check_dir(fig_dir)
    plot_waveforms_par = partial(plot_waveforms,
                                 c['dataset_name'],
                                 c['trace_length'],
                                 c['waveforms_dir'],
                                 fig_dir)
    plot_waveforms_par(event_id)
    print('Plotting waveforms.')
    plot_waveform_wab(fig_dir)

    # cleanup(fig_dir)
