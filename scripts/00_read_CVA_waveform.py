import seisblue

CVA_list = seisblue.utils.get_dir_list('/home/andy/CWB/20*/',
                                       suffix='.txt')
save_dir = '/home/andy/mseed/CVA_TO_MSEED/'

seisblue.io.read_CVA_waveform(CVA_list, save_dir)
