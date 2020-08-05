import seisnn

dataset = seisnn.data.io.read_dataset('HL2017')

for example in dataset:
    instance = seisnn.data.core.Instance(example)
    instance.plot(enlarge=True, snr=True)
