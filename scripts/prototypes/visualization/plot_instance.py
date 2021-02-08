import seisnn

dataset = seisnn.io.read_dataset('HL2017')

for example in dataset:
    instance = seisnn.core.Instance(example)
    instance.plot(enlarge=True, snr=True)
