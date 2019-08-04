import apache_beam as beam
from apache_beam.io import tfrecordio
from apache_beam.options.pipeline_options import PipelineOptions

from seisnn.beam.obspyio import ReadHYP, FeatureToExample, DropEmptyStation, FilterPickPhase, GeneratePDF, GetWindowFromPick, GroupStreamPick, \
    ReadSDS, ReadSfile, StreamFeatureExtraction, TrimTrace

file_dir = '/mnt/Data/test_sfile'
sds_root = '/mnt/DATA'
hyp_file = '/mnt/tf_data/geom/STATION0.HYP'
tfrecord_dir = '/mnt/tf_data/dataset/tfrecord'

with beam.Pipeline(options=PipelineOptions()) as p:
    location = (p
                | 'Get location from HYP' >> ReadHYP(hyp_file)
                )
    picks = (p
             | 'Get events form Sfile' >> ReadSfile(file_dir)
             | 'Get picks from event' >> beam.FlatMap(lambda event: event.picks)
             )

    streams = (picks
               | 'Get only P phase pick' >> beam.ParDo(FilterPickPhase('P'))
               | 'Get window from pick time' >> beam.ParDo(GetWindowFromPick(trace_length=30))
               | 'Get stream from window' >> beam.ParDo(ReadSDS(sds_root=sds_root))
               | 'Remove mean' >> beam.FlatMap(lambda stream: [stream.detrend('demean')])
               | 'Remove trend' >> beam.FlatMap(lambda stream: [stream.detrend('linear')])
               | 'Resample to 100 Hz' >> beam.FlatMap(lambda stream: [stream.resample(100)])
               | 'Trim traces' >> beam.ParDo(TrimTrace(points=3001))
               )

    station_location = location | '(sta, loc)' >> beam.FlatMap(lambda loc: [(loc['station'], loc)])
    station_pick = picks | '(sta, pick)' >> beam.FlatMap(lambda pick: [(pick.waveform_id.station_code, pick)])
    station_stream = streams | '(sta, stream)' >> beam.FlatMap(lambda stream: [(stream[0].stats.station, stream)])

    dataset = ({'pick': station_pick, 'stream': station_stream, 'location': station_location}
               | 'Join by station' >> beam.CoGroupByKey()
               | 'Drop empty station' >> beam.ParDo(DropEmptyStation())
               | 'Group stream pick by time' >> GroupStreamPick()
               | 'Generate stream PDFs' >> beam.ParDo(GeneratePDF(sigma=0.1))
               | 'Extract stream features' >> beam.ParDo(StreamFeatureExtraction())
               )

    _ = (dataset
         | 'Feature to Example' >> beam.ParDo(FeatureToExample())
         # | 'Write dataset' >> tfrecordio.WriteToTFRecord("")
         )
