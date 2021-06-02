
import seisnn


CVA_list = seisnn.utils.get_dir_list('/home/andy/CWB/20*/',
                                     suffix='.txt')
save_dir = '/home/andy/mseed/CVA_TO_MSEED/'

seisnn.io.read_CVA_waveform(CVA_list,save_dir)