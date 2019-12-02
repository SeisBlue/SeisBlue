class LengthError(BaseException):
    pass


def signal_preprocessing(stream):
    stream.detrend('demean')
    stream.detrend('linear')
    stream.normalize()
    stream.resample(100)
    return stream


def trim_trace(stream, points=3001):
    trace = stream[0]
    start_time = trace.stats.starttime
    dt = (trace.stats.endtime - trace.stats.starttime) / (trace.data.size - 1)
    end_time = start_time + dt * (points - 1)
    stream.trim(start_time, end_time, nearest_sample=False, pad=True, fill_value=0)
    return trace

