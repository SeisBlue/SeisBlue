import seisnn

dataset = seisnn.data.io.read_dataset('HL2017')

for batch in dataset.shuffle(1000).batch(2):
    for example in seisnn.data.example_proto.batch_iterator(batch):
        instance = seisnn.data.core.Instance(example)
        instance.plot(enlarge=True, snr=True)
