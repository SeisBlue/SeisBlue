# -*- coding: utf-8 -*-
import os
import glob
import matplotlib.pyplot as plt
from cartopy.io import img_tiles
import cartopy.crs as ccrs
from cartopy.mpl import ticker
from mpl_toolkits.axes_grid1 import axes_size, make_axes_locatable
import matplotlib as mpl
import numpy as np
from obspy.imaging.beachball import beach, beachball
from datetime import datetime
import subprocess
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from math import pi
import datetime
import h5py
import waveform_processing as wf
from itertools import chain
import matplotlib.dates as mdates
from datetime import timedelta
import datetime as dt
import operator
from functools import partial

import bokeh
from bokeh.plotting import figure, show
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, LinearColorMapper, HoverTool, \
    BasicTicker, ColorBar, PrintfTickFormatter, Slider, CustomJS, CheckboxGroup, \
    Circle, Range1d, BoxAnnotation, Select, WMTSTileSource, Spacer, Div, \
    TapTool, DateRangeSlider, Label
from pyproj import CRS, Transformer
from bokeh.transform import transform
from obspy import UTCDateTime
from bokeh.palettes import Turbo256
from bokeh.layouts import gridplot, column
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn

from seisblue import io, core, tool
import inventory_processing as inv


def read_coupe(filename):
    events = []
    with open(filename, 'r') as f:
        with open(filename + '_map', 'r') as fmap:
            line = f.readline()
            line_map = fmap.readline()
            while line:
                e = line.split(' ')
                emap = line_map.split(' ')
                events.append(
                    core.Event(longitude=emap[0], latitude=emap[1], depth=e[2],
                               focal_mechanism=core.NodalPlane(
                                   strike=e[3], dip=e[4], rake=e[5]),
                               magnitude=e[9], time=e[13].strip()))
                line = f.readline()
                line_map = fmap.readline()
    return events


def get_fault_type(rake):
    if -160 < rake < -20:
        type = 'normal'
    elif 20 < rake < 160:
        type = 'reverse'
    else:
        type = 'strike-slip'
    return type


def ax_scatter(
        ax,
        x_data,
        y_data,
        c=None,
        vmax=None,
        event_depth_cmap=mpl.cm.turbo_r,
        event_mark_size=10,
        alpha=0.7,
):
    axes = ax.scatter(
        x_data,
        y_data,
        label="Event",
        c=c,
        cmap=event_depth_cmap,
        vmin=0,
        vmax=vmax,
        edgecolors="k",
        linewidth=0.1,
        marker="o",
        alpha=alpha,
        s=event_mark_size,
    )
    return axes


def ax_scatter_profile(
        ax,
        x_data,
        y_data,
        event_mark_size=6,
        alpha=0.7,
):
    ax.scatter(
        x_data,
        y_data,
        label="Event",
        edgecolors="k",
        linewidth=0.1,
        marker="o",
        alpha=alpha,
        s=event_mark_size,
        c='lightgray'
    )


def transform_projection(events, img_tile):
    eq, focmec = [], []
    eq_sorted = []
    for event in events:
        fm = event.focal_mechanism
        if fm:
            eq.append(
                [float(event.longitude), float(event.latitude),
                 float(event.depth), float(fm.strike), float(fm.dip),
                 float(fm.rake), str(event.time)])
            eq_sorted = sorted(eq, key=lambda x: x[2], reverse=True)
            focmec = np.array(eq_sorted)[:, 3:]
        else:
            eq.append(
                [float(event.longitude), float(event.latitude),
                 float(event.depth)])
            eq_sorted = sorted(eq, key=lambda x: x[2], reverse=True)

    eq_sorted = np.array(eq_sorted).T
    eq_sorted = img_tile.crs.transform_points(
        ccrs.PlateCarree(), eq_sorted[0], eq_sorted[1], eq_sorted[2]
    ).T
    return eq_sorted, focmec


def plot_map(
        geometry,
        network,
        events,
        events_fm,
        right_profile_events=None,
        bottom_profile_events=None,
        center=None,
        pad=None,
        depth_plot=True,
        max_depth=None,
        depth_size=1.5,
        fig_dir='.',
        out_file='map.png',
        type='satellite',
        supplot_pad='-20%',
        beach_size=800,
):
    """
    Plot map.

    :param geometry:
    :param events:
    :param boarder:
    :param depth_plot:
    """
    if type == 'satellite':
        basemap = img_tiles.GoogleTiles(style='satellite',
                                        desired_tile_form='L')
        cmap = 'binary'
    else:
        basemap = img_tiles.OSM(desired_tile_form='L')
        cmap = 'gray'

    fig, ax_map = plt.subplots(
        subplot_kw={"projection": basemap.crs}, figsize=(10, 10)
    )
    ax_map.add_image(basemap, 12, cmap=cmap)

    fault_color = {'reverse': 'r', 'normal': 'b', 'strike-slip': 'g'}
    event_fm_color = {
        event.time: fault_color[get_fault_type(event.focal_mechanism.rake)] for
        event in events_fm
    }

    if events:
        eq, _ = transform_projection(events, basemap)
        if max_depth is None:
            max_depth = max(eq[2])
        im = ax_scatter(ax_map, eq[0], eq[1], c=eq[2], vmax=max_depth)

    if events_fm:
        eq_fm, focmec = transform_projection(events_fm, basemap)
        profile_fm_color = [event_fm_color[f[-1]] for f in focmec]
        max_depth = max(eq_fm[2])
        xmin, ymin = min(eq_fm[0]), min(eq_fm[1])
        xmax, ymax = max(eq_fm[0]), max(eq_fm[1])
        # ax_map.set_xlim(xmin, xmax)
        # ax_map.set_ylim(ymin, ymax)

        eq_fm = eq_fm.T
        for i, e in enumerate(eq_fm):
            fault_type = get_fault_type(focmec[i][2])
            color = fault_color[fault_type]
            b = beach(focmec[i, :3], xy=(e[0], e[1]), width=beach_size + 200,
                      facecolor=color,
                      linewidth=0.2)
            b.set_zorder(10)
            ax = plt.gca()
            ax.add_collection(b)
            zoom_in_depth = max(eq_fm[2])
        proxy = [Patch(facecolor=color, label=fault_type) for fault_type, color
                 in fault_color.items()]
    if right_profile_events:
        eq_fm_right, focmec_right = transform_projection(right_profile_events,
                                                         basemap)
        eq_fm_right[2] = eq_fm_right[2] * 1000
        eq_fm_right = eq_fm_right.T
    if bottom_profile_events:
        eq_fm_bottom, focmec_bottom = transform_projection(
            bottom_profile_events,
            basemap)
        eq_fm_bottom[2] = eq_fm_bottom[2] * 1000
        eq_fm_bottom = eq_fm_bottom.T
    if geometry:
        geom = []
        for station in geometry:
            geom.append([station.longitude, station.latitude])
        geom = np.array(geom).T
        geom = basemap.crs.transform_points(
            ccrs.PlateCarree(), geom[0], geom[1]
        ).T
        ax_map.scatter(
            geom[0],
            geom[1],
            label=network,
            color="k",
            edgecolors="k",
            linewidth=0.3,
            marker="^",

            s=40,
        )

    # PlateCarree
    xmin, xmax = ax_map.get_xlim()
    ymin, ymax = ax_map.get_ylim()

    # convert to Google Mercator (lat lon)
    conv = ProjectionConverter(basemap.crs, ccrs.PlateCarree())
    xmin, ymin = conv.convert(xmin, ymin)
    xmax, ymax = conv.convert(xmax, ymax)
    if xmax - xmin > ymax - ymin:
        ymin = (ymax + ymin) / 2 - (xmax - xmin) / 2
        ymax = (ymax + ymin) / 2 + (xmax - xmin) / 2
    else:
        xmin = (xmax + xmin) / 2 - (ymax - ymin) / 2
        xmax = (xmax + xmin) / 2 + (ymax - ymin) / 2

    center = [(xmax + xmin) / 2, (ymax + ymin) / 2] if not center else center
    if pad:
        xmin, xmax, ymin, ymax = [
            center[0] - pad,
            center[0] + pad,
            center[1] - pad,
            center[1] + pad,
        ]

    # colorbar
    cbaxes = inset_axes(ax_map, width="1%", height="30%", loc=3,
                        bbox_to_anchor=(0.01, 0.05, 1, 1),
                        bbox_transform=ax_map.transAxes)
    cb = plt.colorbar(im, cax=cbaxes)
    cb.ax.set_title('km', fontsize=10)
    cb.ax.invert_yaxis()
    formatter = mpl.ticker.FuncFormatter(lambda x, _: f'{x / 1000:.1f}')
    cb.ax.yaxis.set_major_formatter(formatter)

    # ticks
    xticks = ticker.LongitudeLocator(nbins=5)._raw_ticks(xmin, xmax)
    yticks = ticker.LatitudeLocator(nbins=5)._raw_ticks(ymin, ymax)

    # convert to PlateCarree
    ax_map.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax_map.set_yticks(yticks, crs=ccrs.PlateCarree())

    ax_map.xaxis.set_major_formatter(
        ticker.LongitudeFormatter(zero_direction_label=True)
    )
    ax_map.yaxis.set_major_formatter(ticker.LatitudeFormatter())

    ax_map.xaxis.set_ticks_position("both")
    ax_map.yaxis.set_ticks_position("both")
    ax_map.legend(loc='upper right')
    ax_map.set_title(
        f'{len(events)} events ({len(events_fm)} with fm), {len(geometry)} stations')

    # convert PlateCarree
    conv = ProjectionConverter(ccrs.PlateCarree(), basemap.crs)
    xmin, ymin = conv.convert(min(xticks), min(yticks))
    xmax, ymax = conv.convert(max(xticks), max(yticks))

    ax_map.set_xlim(xmin, xmax)
    ax_map.set_ylim(ymin, ymax)
    divider = make_axes_locatable(ax_map)

    # scalebar
    length = 5
    latitude_rad = math.radians(np.mean(yticks))
    lon_diff = length / (math.pi / 180 * 6371 * math.cos(latitude_rad))
    x1, y1 = conv.convert(min(xticks) + lon_diff, min(yticks))
    step = (x1 - xmin) / length
    for i in range(length):
        color = 'black' if i % 2 == 0 else 'white'
        x0 = xmin + step * i
        x1 = xmin + step * (i + 1)
        ax_map.hlines(ymin, x0, x1, transform=ax_map.projection,
                      color=color, linewidth=5)
    ax_map.text((xmin + x1) / 2, ymin, f'{length} km',
                ha='center', va='bottom')

    if events and depth_plot:
        # right depth plot
        ax_lat_dep = divider.append_axes(
            "right",
            "100%",
            pad=supplot_pad,
            sharey=ax_map,
            projection=basemap.crs,
        )
        ax_lat_dep.legend(handles=proxy, loc='upper right')
        ax_lat_dep.set_aspect((1 / depth_size))
        ax_scatter_profile(ax_lat_dep, eq[2], eq[1])
        for i, e in enumerate(eq_fm_right):
            b = beach(list(map(float, focmec_right[i, :3])),
                      xy=(e[2], e[1]),
                      width=beach_size,
                      facecolor=profile_fm_color[i],
                      linewidth=0.2)
            b.set_zorder(10)
            ax_lat_dep.add_collection(b)

        step = max_depth / 4
        step = np.ceil(
            step / (10 ** np.floor(np.log10(step)))) * 10 ** np.floor(
            np.log10(step)
        )
        ax_lat_dep.set_xticks([0, step * 1, step * 2, step * 3, step * 4])
        ax_lat_dep.set_xticklabels(
            [
                0,
                int(step / 1000),
                int(step * 2 / 1000),
                int(step * 3 / 1000),
                int(step * 4 / 1000),
            ]
        )
        ax_lat_dep.set_aspect('equal')
        ax_lat_dep.set_xlim(0, step * 4)
        ax_lat_dep.set_yticks(yticks, crs=ccrs.PlateCarree())
        ax_lat_dep.yaxis.tick_right()
        ax_lat_dep.yaxis.set_ticks_position("both")
        ax_lat_dep.yaxis.set_tick_params(labelleft=False)
        ax_lat_dep.set_xlabel('(km)')

        # bottom depth plot
        ax_lon_dep = divider.append_axes(
            "bottom",
            "100%",
            pad=supplot_pad,
            sharex=ax_map,
            projection=basemap.crs,
        )
        ax_lon_dep.set_aspect(depth_size)
        ax_lon_dep.set_aspect('equal')
        ax_lon_dep.set_xlim(xmin, xmax)
        ax_scatter_profile(ax_lon_dep, eq[0], eq[2])

        for i, e in enumerate(eq_fm_bottom):
            b = beach(list(map(float, focmec_bottom[i, :3])),
                      xy=(e[0], e[2]),
                      width=beach_size,
                      facecolor=profile_fm_color[i],
                      linewidth=0.2)
            b.set_zorder(10)
            ax_lon_dep.add_collection(b)

        ax_lon_dep.set_yticks([0, step * 1, step * 2, step * 3, step * 4])
        ax_lon_dep.set_yticklabels(
            [
                0,
                int(step / 1000),
                int(step * 2 / 1000),
                int(step * 3 / 1000),
                int(step * 4 / 1000),
            ]
        )
        ax_lon_dep.set_ylim(0, step * 4)
        ax_lon_dep.xaxis.set_ticks_position("both")
        ax_lon_dep.set_xticks(xticks, crs=ccrs.PlateCarree())
        ax_lon_dep.invert_yaxis()
        ax_lon_dep.set_ylabel('(km)')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, out_file), dpi=300)

    # plt.show()


def get_event(obspy_event):
    """
    Returns event objects from obspy events .
    :param list[obspy.core.event.event.Event] obspy_events: List of obspy events.
    :rtype: list[core.Event]
    :return: List of event objects.
    """
    origin_time = obspy_event.origins[0].time
    latitude = obspy_event.origins[0].latitude
    longitude = obspy_event.origins[0].longitude
    depth = obspy_event.origins[0].depth
    # magnitude = obspy_event.magnitudes[0].mag
    if len(obspy_event.focal_mechanisms) > 0:
        np1 = obspy_event.focal_mechanisms[0].nodal_planes.nodal_plane_1
        np = core.NodalPlane(strike=np1.strike,
                             strike_errors=np1.strike_errors.uncertainty,
                             dip=np1.dip,
                             dip_errors=np1.dip_errors.uncertainty,
                             rake=np1.rake,
                             rake_errors=np1.rake_errors.uncertainty)
    else:
        np = None

    event = core.Event(
        time=datetime.utcfromtimestamp(origin_time.timestamp),
        latitude=latitude,
        longitude=longitude,
        depth=depth,
        focal_mechanism=np)

    return event


class ProjectionConverter:
    """
    Cartopy projection converter.
    """

    def __init__(self, source_proj, target_proj):
        self.x = None
        self.y = None
        self.source_proj = source_proj
        self.target_proj = target_proj

    def convert(self, x, y):
        """
        Returns converted project location.

        :param float x: X location.
        :param float y: Y location.
        :rtype: float
        :return: (x, y)
        """
        self.x = x
        self.y = y
        result = self.target_proj._project_point(self, self.source_proj)
        return result.x, result.y


def plot_subplot_elements(p, events, events_fm, geometry,
                          transformer, fig_dir,
                          p_right, p_bottom,
                          slider_gap_azi, slider_gap_ain, checkbox_group,
                          slider_ball,
                          select, tile_renderer, tile_sources,
                          ball_size=500,
                          offset_ratio=800):
    if geometry:
        geom = []
        for station in geometry:
            mx, my = transformer.transform(float(station.longitude),
                                           float(station.latitude))
            geom.append([station.station, station.network, mx, my,
                         float(station.longitude), float(station.latitude),
                         float(station.elevation),
                         station.timewindow.starttime.strftime("%Y/%m/%d"),
                         station.timewindow.endtime.strftime("%Y/%m/%d")])
        stations, networks, x, y, lons, lats, elevs, stimes, etimes = zip(*geom)

        station_source = ColumnDataSource(dict(x=x, y=y, lon=lons, lat=lats,
                                               station=stations,
                                               network=networks,
                                               elevation=elevs,
                                               starttime=stimes,
                                               endtime=etimes))
        geom_renderer = p.triangle(x="x", y="y", size=8, fill_color="black",
                                   line_color='black',
                                   source=station_source,
                                   legend_label='network')
        sthover = HoverTool(renderers=[geom_renderer])
        sthover.tooltips = """
            <div>
                <div><strong>Station:</strong> @station</div>
                <div><strong>Network:</strong> @network</div>
                <div><strong>Longitude:</strong> @lon</div>
                <div><strong>Latitude:</strong> @lat</div>
                <div><strong>Elevation:</strong> @elevation</div>
                <div><strong>StartTime:</strong> @starttime</div>
                <div><strong>EndTime:</strong> @endtime</div>
            </div>
        """
        p.add_tools(sthover)
        geom_r_renderer = p_right.triangle(x=-1, y="y", size=6,
                                           fill_color="black",
                                           line_color='black', angle=pi / 2,
                                           source=station_source)
        geom_b_renderer = p_bottom.triangle(x="x", y=-1, size=6,
                                            fill_color="black",
                                            line_color='black',
                                            source=station_source)

    # events
    eq = []
    datetimes = []

    for event in events:
        mx, my = transformer.transform(float(event.longitude),
                                       float(event.latitude))

        datetimes.append(event.time.datetime)
        date = str(event.time.datetime.date())
        time = str(event.time)

        eq.append((time, mx, my, float(event.depth) / 1000,
                   float(event.longitude), float(event.latitude), date))

    time, x, y, depths, lons, lats, dates = zip(*eq)
    events_source = ColumnDataSource(
        data=dict(time=time, x=x, y=y, depth=depths,
                  lon=lons, lat=lats, original_x=x, original_y=y, date=dates))

    color_mapper = LinearColorMapper(palette=Turbo256,
                                     low=max(depths),
                                     high=min(depths))
    events_renderer = p.circle(x="x", y="y",
                               size=3,
                               color=transform('depth', color_mapper),
                               fill_alpha=0.8,
                               source=events_source,
                               legend_label='events')
    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12,
                         location=(0, 0),
                         ticker=BasicTicker(),
                         formatter=PrintfTickFormatter(format="%d km"),
                         width=8,
                         height=100,
                         background_fill_color=None,
                         background_fill_alpha=0
                         )
    p.add_layout(color_bar, 'left')
    time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    p.title.text = f"Last Updated: {time_now}"
    p.title.align = "right"

    events_r_renderer = p_right.circle(x="depth", y="y",
                                       size=3,
                                       color='gray',
                                       fill_alpha=0.7,
                                       line_color=None,
                                       source=events_source)
    events_b_renderer = p_bottom.circle(x="x", y="depth",
                                        size=3,
                                        color='gray',
                                        fill_alpha=0.5,
                                        line_color=None,
                                        source=events_source)
    p_right.x_range.start = -2
    p_right.x_range.end = max(depths) + 2
    p_bottom.y_range.start = max(depths) + 2
    p_bottom.y_range.end = -2
    slider_date = DateRangeSlider(title="Date Range",
                                  start=min(datetimes),
                                  end=max(datetimes), value=(
            min(datetimes), max(datetimes)), step=1)

    hover = HoverTool(renderers=[events_renderer, events_r_renderer, events_b_renderer])
    hover.tooltips = """
        <div>
            <div><strong>Time:</strong> @time</div>
            <div><strong>Longitude:</strong> @lon</div>
            <div><strong>Latitude:</strong> @lat</div>
            <div><strong>Depth:</strong> @depth</div>
        </div>
    """
    p.add_tools(hover)
    p_right.add_tools(hover)
    p_bottom.add_tools(hover)


    # events_fm:
    fault_color = {'reverse': 'red', 'normal': 'blue',
                   'strike-slip': 'green'}

    eq = []
    for event in events_fm:
        mx, my = transformer.transform(float(event.origins[0].longitude),
                                       float(event.origins[0].latitude))
        time = UTCDateTime(event.origins[0].time)
        polarity_fig = os.path.join(fig_dir, 'beach_polarity',
                                    f'{time}.jpg')
        beach_fig = os.path.join(fig_dir, 'beach', 'main_map',
                                 f'{time}.png')
        rprofile_fig = os.path.join(fig_dir, 'beach', 'right_profile',
                                    f'{time}.png')
        bprofile_fig = os.path.join(fig_dir, 'beach', 'bottom_profile',
                                    f'{time}.png')
        date = str(time.datetime.date())
        gap_azi = event.origins[0].quality.azimuthal_gap
        ains = []
        for pick in event.picks:
            if pick.phase_hint == 'P':
                arrival = \
                    [arr for arr in event.origins[0].arrivals if
                     arr.pick_id == pick.resource_id][0]
                ains.append(int(arrival.takeoff_angle))
        gap_ain = max(ains) - min(ains)

        eq.append((str(time), mx, my, float(event.origins[0].depth) / 1000,
                   float(event.origins[0].longitude),
                   float(event.origins[0].latitude),
                   beach_fig, polarity_fig, gap_azi, gap_ain,
                   rprofile_fig, bprofile_fig, date))

    time, x, y, depths, lons, lats, beach_urls, polarity_urls, gap_azi, gap_ain, rprofile_fig, bprofile_fig, dates = zip(
        *eq)
    w = [ball_size] * len(events_fm)
    h = [ball_size] * len(events_fm)

    w_depth = [ball_size / offset_ratio] * len(events_fm)
    h_depth = [ball_size / offset_ratio] * len(events_fm)
    source = ColumnDataSource(
        data=dict(time=time, x=x, y=y, depth=depths,
                  lon=lons, lat=lats,
                  beach_url=beach_urls, polarity_url=polarity_urls,
                  w=w, h=h,
                  gap_azi=gap_azi, gap_ain=gap_ain,
                  original_x=x, original_y=y,
                  rprofile_url=rprofile_fig, bprofile_url=bprofile_fig,
                  w_depth=w_depth, h_depth=h_depth,
                  date=dates,
                  original_w=w, original_h=h))
    source_table = ColumnDataSource(
        data=dict(time=time, x=x, y=y, depth=depths,
                  lon=lons, lat=lats,
                  beach_url=beach_urls, polarity_url=polarity_urls,
                  w=w, h=h,
                  gap_azi=gap_azi, gap_ain=gap_ain,
                  original_x=x, original_y=y,
                  rprofile_url=rprofile_fig, bprofile_url=bprofile_fig,
                  w_depth=w_depth, h_depth=h_depth,
                  date=dates))
    fm_legend = []
    for type, color in fault_color.items():
        fm_legend.append(p.rect(x=x[0], y=y[0], width=0.2, height=1,
                                fill_color=color,
                                line_color='black',
                                source=source, legend_label=type, alpha=1))
    events_fm_renderer = p.circle(x="x", y="y",
                                  size=15,
                                  color='red',
                                  fill_alpha=0,
                                  line_color=None,
                                  source=source_table)

    callback = CustomJS(args=dict(source=source, p=p, offset_ratio=offset_ratio, slider=slider_ball), code="""
        var data = source.data;
        var offset_ratio = offset_ratio;
        var value = slider.value;
 
        var x_range = p.x_range;
        var y_range = p.y_range;
        var size = value * Math.min(Math.abs(x_range.end - x_range.start), Math.abs(y_range.end - y_range.start)) / 100;
        
        for (var i = 0; i < data['beach_url'].length; i++) {
            data['w'][i] = size;
            data['h'][i] = size;
        }
    
        source.change.emit();
    """)
    p.x_range.js_on_change('start', callback)
    p.x_range.js_on_change('end', callback)
    p.y_range.js_on_change('start', callback)
    p.y_range.js_on_change('end', callback)
    slider_ball.js_on_change('value', callback)

    nonselected_circle = Circle(fill_alpha=0, line_color=None)
    selected_circle = Circle(fill_alpha=0.8, line_color=None, fill_color='red')
    events_fm_renderer.nonselection_glyph = nonselected_circle
    events_fm_renderer.selection_glyph = selected_circle

    fm_renderer = p.image_url(url='beach_url', x='x', y='y',
                              anchor="center",
                              source=source, w='w', h='h')

    # right
    events_fm_r_renderer = p_right.circle(x="depth", y="y",
                                          size=12,
                                          fill_color='red',
                                          fill_alpha=0,
                                          line_color=None,
                                          source=source_table)
    events_fm_r_renderer.nonselection_glyph = nonselected_circle
    events_fm_r_renderer.selection_glyph = selected_circle
    fm_r_renderer = p_right.image_url(url='rprofile_url', x='depth', y='y',
                                      anchor="center",
                                      source=source, w='w_depth', h='original_h')

    # bottom
    events_fm_b_renderer = p_bottom.circle(x="x", y="depth",
                                           size=12,
                                           fill_color='red',
                                           fill_alpha=0,
                                           line_color=None,
                                           source=source_table)
    events_fm_b_renderer.nonselection_glyph = nonselected_circle
    events_fm_b_renderer.selection_glyph = selected_circle
    fm_b_renderer = p_bottom.image_url(url='bprofile_url', x='x', y='depth',
                                       anchor="center",
                                       source=source, w='original_w', h='h_depth')

    callback = CustomJS(args=dict(source=source, events_source=events_source,
                                  slider_azi=slider_gap_azi,
                                  slider_ain=slider_gap_ain,
                                  slider_date=slider_date), code="""
        var data = source.data;
        var data_circle = events_source.data;
        var f_azi = slider_azi.value;
        var f_ain = slider_ain.value;
        var gap_azi = data['gap_azi'];
        var gap_ain = data['gap_ain'];
        var original_x = data['original_x'];
        var original_y = data['original_y'];
        var x = data['x'];
        var y = data['y'];

        var original_cx = data_circle['original_x'];
        var original_cy = data_circle['original_y'];
        var cx = data_circle['x'];
        var cy = data_circle['y'];

        var dates = data['date'];
        var cdates = data_circle['date'];

        var range_start = new Date(slider_date.value[0]);
        var range_end = new Date(slider_date.value[1]);

        for (var i = 0; i < x.length; i++) {
             var date = new Date(Date.parse(dates[i]))
             if (gap_azi[i] > f_azi || gap_ain[i] < f_ain || date < range_start || date > range_end ) {
                 x[i] = NaN;
                 y[i] = NaN;
             } else {
                 x[i] = original_x[i];
                 y[i] = original_y[i];
             }
        }
        for (var i = 0; i < cx.length; i++) {
             var cdate = new Date(Date.parse(cdates[i]))
             if (cdate < range_start || cdate > range_end ) {
                 cx[i] = NaN;
                 cy[i] = NaN;
             } else {
                 cx[i] = original_cx[i];
                 cy[i] = original_cy[i];
             }
        }
        source.change.emit();
        events_source.change.emit();
    """)
    slider_gap_azi.js_on_change('value', callback)
    slider_gap_ain.js_on_change('value', callback)
    slider_date.js_on_change('value', callback)

    hover = HoverTool(renderers=[events_fm_renderer, events_fm_b_renderer,
                                 events_fm_r_renderer])
    hover.tooltips = """
        <div>
            <div><strong>Time:</strong> @time</div>
            <div><strong>Longitude:</strong> @lon</div>
            <div><strong>Latitude:</strong> @lat</div>
            <div><strong>Depth:</strong> @depth</div>
            <div><img src="@polarity_url" alt="" width="150" /></div>
        </div>
    """
    p.add_tools(hover)
    p_right.add_tools(hover)
    p_bottom.add_tools(hover)

    callback = CustomJS(args=dict(circle_renderer=events_renderer,
                                  image_renderer=fm_renderer,
                                  image_r_renderer=fm_r_renderer,
                                  image_b_renderer=fm_b_renderer,
                                  circle_r_renderer=events_r_renderer,
                                  circle_b_renderer=events_b_renderer,
                                  triangle_renderer=geom_renderer,
                                  triangle_r_renderer=geom_r_renderer,
                                  triangle_b_renderer=geom_b_renderer,
                                  fm_legend=fm_legend,
                                  circle_table_render=events_fm_renderer,
                                  circle_table_r_render=events_fm_r_renderer,
                                  circle_table_b_render=events_fm_b_renderer), code="""
        circle_renderer.visible = cb_obj.active.includes(0);
        circle_r_renderer.visible = cb_obj.active.includes(0);
        circle_b_renderer.visible = cb_obj.active.includes(0);
        image_renderer.visible = cb_obj.active.includes(1);
        image_r_renderer.visible = cb_obj.active.includes(1);
        image_b_renderer.visible = cb_obj.active.includes(1);
        circle_table_render.visible = cb_obj.active.includes(1);
        circle_table_r_render.visible = cb_obj.active.includes(1);
        circle_table_b_render.visible = cb_obj.active.includes(1);
        triangle_renderer.visible = cb_obj.active.includes(2);
        triangle_r_renderer.visible = cb_obj.active.includes(2);
        triangle_b_renderer.visible = cb_obj.active.includes(2);
        for (var i = 0; i < fm_legend.length; i++) {
            fm_legend[i] = cb_obj.active.includes(1);
        }
    """)
    checkbox_group.js_on_change('active', callback)

    # tile map menu
    callback = CustomJS(
        args=dict(tile_renderer=tile_renderer, tile_sources=tile_sources), code="""
        tile_renderer.tile_source = tile_sources[cb_obj.value];
    """)

    select.js_on_change('value', callback)

    # table
    columns = [
        TableColumn(field="time", title="Time"),
        TableColumn(field="lon", title="Longitude"),
        TableColumn(field="lat", title="Latitude"),
        TableColumn(field="depth", title="Depth")
    ]
    data_table = DataTable(source=source_table, columns=columns, width=500,
                           height=250, selectable=True)

    return slider_date, data_table


def plot_map_web(events, events_fm, geometry, fig_dir, outputfile="gmap.html"):
    tile_sources = {
        "OpenStreetMap": WMTSTileSource(
            url="https://c.tile.openstreetmap.org/{Z}/{X}/{Y}.png"),
        "CartoDB Positron": WMTSTileSource(
            url="https://basemaps.cartocdn.com/light_all/{Z}/{X}/{Y}.png"),
        "Esri World Imagery": WMTSTileSource(
            url='https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{Z}/{Y}/{X}.png')
    }
    output_file(outputfile)
    p_main = figure(x_axis_type="mercator", y_axis_type="mercator", height=550,
                    match_aspect=True)
    tile_renderer = p_main.add_tile(tile_sources["OpenStreetMap"], retina=True)
    p_right = figure(y_axis_type="mercator", match_aspect=True, y_range=p_main.y_range,
                     width=220, height=p_main.height)
    p_bottom = figure(x_axis_type="mercator", match_aspect=True, x_range=p_main.x_range,
                      width=p_main.width, height=200)
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857",
                                       always_xy=True)

    slider_gap_azi = Slider(start=0, end=360, value=360, step=5,
                            title="gap_azi_threshold")
    slider_gap_ain = Slider(start=0, end=180, value=0, step=5,
                            title="gap_ain_threshold")
    checkbox_group = CheckboxGroup(
        labels=["Events", "FocalMechanism", "Stations"],
        active=[0, 1, 2])

    select = Select(title="Tile Map:", value="OpenStreetMap",
                    options=list(tile_sources.keys()))
    slider_ball = Slider(start=1, end=10, value=3, step=1,
                         title="BeachBall size")

    slider_date, data_table = plot_subplot_elements(p_main, events, events_fm,
                                                    geometry,
                                                    transformer, fig_dir,
                                                    p_right, p_bottom,
                                                    slider_gap_azi,
                                                    slider_gap_ain,
                                                    checkbox_group,
                                                    slider_ball,
                                                    select, tile_renderer,
                                                    tile_sources,
                                                    offset_ratio=800)

    spacer_middle = Spacer(height=100)

    controls_column = column(select, slider_date, slider_ball, checkbox_group,
                             spacer_middle,
                             slider_gap_azi, slider_gap_ain)
    spacer_left = Spacer(width=150)

    layout = gridplot(
        [[spacer_left, p_main, p_right, controls_column],
         [None, p_bottom, None, data_table]],
        toolbar_location="above")
    show(layout)


if __name__ == '__main__':
    config = io.read_yaml('./config/data_config.yaml')
    c = config['plot_map']

    event_files = sorted(glob.glob(f'./dataset/{c["dataset_name"]}*test*.hdf5'))
    event_instances = tool.parallel(event_files,
                                    func=io.read_hdf5_event)
    events = [event_instance.event for sub_list in event_instances for
              event_instance in sub_list]
    print(f"Get {len(events)} events from {c['dataset_name']}\n{event_files}.")

    station_time_dict = inv.get_stations_time_window(c['sub_waveforms_dir'])
    print(
        f'Get {len(station_time_dict)} stations with time window from {c["sub_waveforms_dir"]}')
    geom = inv.read_hyp(c['hyp_filepath'], station_time_dict, c['network'])

    events_dir = glob.glob(f"./result/{c['dataset_name']}/fine")[0]
    obspy_events_fm = io.get_obspy_events(events_dir)

    print(
        f"Get {len(obspy_events_fm)} events from {c['dataset_name']} with good focal mechanism.")

    print('Plot map.')
    fig_dir = os.path.join(f"./figure/{c['dataset_name']}")
    # plot_map(geom,
    #          c['network'],
    #          events=events,
    #          events_fm=obspy_events_fm,
    #          right_profile_events=right_profile_events,
    #          bottom_profile_events=bottom_profile_events,
    #          fig_dir=fig_dir,
    #          out_file='map_street.jpg',
    #          pad=0.1,
    #          type='street',
    #          supplot_pad='-20%',
    #          beach_size=400)

    plot_map_web(events, obspy_events_fm, geom, fig_dir, c['outputfile'])
