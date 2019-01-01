import scipy.stats as ss


def get_probability(stream):
    for trace in stream:
        start_time = trace.meta.starttime
        x_time = trace.times(reftime=start_time)
        pick_time = trace.pick.time - start_time
        sigma = 0.1

        pdf = ss.norm.pdf(x_time, pick_time, sigma)
        if pdf.max():
            trace.pick.pdf = pdf / pdf.max()
        else:
            stream.remove(trace)
    return stream


def set_probability(stream, predict):
    trace_length = stream.traces[0].data.size
    predict = predict.reshape(len(stream), trace_length)

    i = 0
    for trace in stream:
        trace.pick.pdf = predict[i, :]
        i += 1
    return stream
