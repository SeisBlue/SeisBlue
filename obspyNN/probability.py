import scipy.stats as ss
from obspy.core.event.origin import Pick


def get_probability(stream):
    for trace in stream:
        start_time = trace.stats.starttime
        x_time = trace.times(reftime=start_time)
        pick_time = trace.picks[0].time - start_time
        sigma = 0.1

        pdf = ss.norm.pdf(x_time, pick_time, sigma)
        if pdf.max():
            trace.picks[0].pdf = pdf / pdf.max()
        else:
            stream.remove(trace)
    return stream


def set_probability(stream, predict):
    trace_length = stream.traces[0].data.size
    predict = predict.reshape(len(stream), trace_length)

    i = 0
    for trace in stream:
        trace.picks[0].pdf = predict[i, :]
        i += 1
    return stream


def extract_picks(trace):
    start_time = trace.stats.starttime
    picks = []
    pick = Pick()
    # TODO: Extract picks from pdf
    pick.time = start_time
    pick.phase_hint = "P"

    picks.append(pick)
    return picks
