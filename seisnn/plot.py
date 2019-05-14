import os

import matplotlib.pyplot as plt
import numpy as np

from seisnn.pick import get_picks_from_pkl


def plot_trace(trace, enlarge=False, xlim=None, save_dir=None):
    start_time = trace.stats.starttime
    time_stamp = start_time.isoformat()

    if trace.picks:
        first_pick_time = trace.picks[0].time - start_time
        pick_phase = trace.picks[0].phase_hint
    else:
        first_pick_time = 1
        pick_phase = ""

    subplot = 2
    fig = plt.figure(figsize=(8, subplot * 2))

    ax = fig.add_subplot(subplot, 1, 1)
    plt.title(time_stamp + " " + trace.id)
    if xlim:
        plt.xlim(xlim)
    if enlarge:
        plt.xlim((first_pick_time - 1, first_pick_time + 2))
    ax.plot(trace.times(reftime=start_time), trace.data, "k-", label=trace.id)
    y_min, y_max = ax.get_ylim()

    if trace.picks:
        val_label = True
        pre_label = True
        for pick in trace.picks:
            pick_time = pick.time - start_time
            if pick.evaluation_mode == "manual":
                if val_label:
                    ax.vlines(pick_time, y_min, y_max, color='g', lw=2,
                              label=pick.evaluation_mode + " " + pick.phase_hint)
                    val_label = False
                else:
                    ax.vlines(pick_time, y_min, y_max, color='g', lw=2)

            elif pick.evaluation_mode == "automatic":
                if pre_label:
                    ax.vlines(pick_time, y_min, y_max, color='r', lw=1,
                              label=pick.evaluation_mode + " " + pick.phase_hint)
                    pre_label = False
                else:
                    ax.vlines(pick_time, y_min, y_max, color='r', lw=1)
    ax.legend(loc=1)

    ax = fig.add_subplot(subplot, 1, subplot)
    ax.plot(trace.times(reftime=start_time), trace.pdf, "b-", label=pick_phase + " pdf")
    if xlim:
        plt.xlim(xlim)
    if enlarge:
        plt.xlim((first_pick_time - 1, first_pick_time + 2))
    ax.legend()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir + "/" + time_stamp + "_" + trace.id + ".pdf")
        plt.close()
    else:
        plt.show()


def plot_error_distribution(predict_pkl_list):
    time_residuals = []
    for i, pkl in enumerate(predict_pkl_list):
        picks = get_picks_from_pkl(pkl)
        for p in picks:
            if p.time_errors:
                residual = p.time_errors.get("uncertainty")
                time_residuals.append(float(residual))

        if i % 1000 == 0:
            print("Reading... %d out of %d " % (i, len(predict_pkl_list)))

    bins = np.linspace(-0.5, 0.5, 100)
    plt.hist(time_residuals, bins=bins)
    plt.xticks(np.arange(-0.5, 0.51, step=0.1))
    plt.xlabel("Time residuals (sec)")
    plt.ylabel("Number of picks")
    plt.title("Error Distribution")
    plt.show()
