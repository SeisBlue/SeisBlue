import seisblue

config = seisblue.utils.Config()

tfr_list = seisblue.utils.get_dir_list('/home/andy/TFRecord/Eval/demo_gan.h5',
                                       suffix='.tfrecord')
dataset = seisblue.io.read_dataset(tfr_list)
for item in dataset:
    instance = seisblue.core.Instance(item)
    instance.plot(threshold=0.4)

# psnr,ssnr = seisblue.qc.get_snr_list(dataset)
# seisblue.plot.plot_snr_distribution(psnr)
# seisblue.plot.plot_snr_distribution(ssnr)
