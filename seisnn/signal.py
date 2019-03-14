class LengthError(BaseException):
    pass


def signal_preprocessing(data):
    data.detrend('demean')
    data.detrend('linear')
    data.normalize()
    data.resample(100)
    return data


def trim_trace(trace, points=3001):
    start_time = trace.stats.starttime
    dt = (trace.stats.endtime - trace.stats.starttime) / (trace.data.size - 1)
    end_time = start_time + dt * (points - 1)
    trace.trim(start_time, end_time, nearest_sample=False, pad=True, fill_value=0)
    if not trace.data.size == points:
        raise LengthError("Trace length is not correct.")
    return trace

