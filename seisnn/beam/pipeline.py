import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from seisnn.beam.obspyio import FilterPickPhase, GetWindowFromPick, PickFeatureExtraction, ReadSDS, ReadSfile, \
    TraceFeatureExtraction, TrimTrace

file_dir = '/mnt/Data/test_sfile'
sds_root = '/mnt/DATA'

with beam.Pipeline(options=PipelineOptions()) as p:
    picks = (p
             | 'Get events form Sfile' >> ReadSfile(file_dir)
             | 'Get picks from event' >> beam.FlatMap(lambda event: event.picks)
             )

    traces = (picks
              | 'Get only P phase pick' >> beam.ParDo(FilterPickPhase('P'))
              | 'Get window from pick time' >> beam.ParDo(GetWindowFromPick(trace_length=30))
              | 'Get trace from window' >> beam.ParDo(ReadSDS(sds_root=sds_root))
              | 'Remove trace mean' >> beam.FlatMap(lambda trace: [trace.detrend('demean')])
              | 'Remove trace trend' >> beam.FlatMap(lambda trace: [trace.detrend('linear')])
              | 'Trace Normalize' >> beam.FlatMap(lambda trace: [trace.normalize()])
              | 'Trace resample to 100 Hz' >> beam.FlatMap(lambda trace: [trace.resample(100)])
              | 'Trim traces' >> beam.ParDo(TrimTrace(points=3001))
              )

    station_pick = (picks
                    | 'Extract pick feature' >> beam.ParDo(PickFeatureExtraction())
                    | 'Set pick key: station' >> beam.FlatMap(lambda x: [(x['station'], x)])
                    )
    station_trace = (traces
                     | 'Extract trace feature' >> beam.ParDo(TraceFeatureExtraction())
                     | 'Set trace key: station' >> beam.FlatMap(lambda x: [(x['station'], x)])
                     )

    results = (
            {'pick': station_pick, 'trace': station_trace}
            | beam.CoGroupByKey()
            | beam.FlatMap(lambda x: print(x))
    )
