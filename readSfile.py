import obspy.io.nordic.core as nordic
from obspy.core import *
import subprocess

sfileDir = "REA/EVENT/1996/06/"
waveFileDir = "WAV/"
lsOutput = subprocess.run(["ls", sfileDir], stdout=subprocess.PIPE, universal_newlines=True)
sfileList = lsOutput.stdout.splitlines()
for file in sfileList:
    event = nordic.read_nordic(sfileDir + file)
    event.wavename = nordic.readwavename(sfileDir + file)
    stream = Stream()
    for wave in event.wavename:
        stream += read(str(waveFileDir + wave))
    stream.normalize()
    start_time = event.events[0].origins[0].time
    stream.trim(start_time+0, start_time + 800)
    stream.plot()
