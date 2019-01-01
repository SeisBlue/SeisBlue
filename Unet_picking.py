from tensorflow import keras
import obspyNN
from obspyNN.model import unet

tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
                                          write_graph=True, write_images=False)
wavedir = "/mnt/Data"
sfileList = obspyNN.io.read_list("/mnt/tf_data/sfilelist")

dataset = obspyNN.io.load_dataset(sfileList, wavedir, plot=False)
wavefile, probability = obspyNN.io.load_training_set(dataset)

split_dataset = -20
model = unet()
model.fit(wavefile[:split_dataset], probability[:split_dataset], epochs=5, callbacks=[tensorboard])

test_loss, test_acc = model.evaluate(wavefile, probability)
print('Test accuracy:', test_acc)

predict = model.predict(wavefile)
pdf = predict.reshape(len(dataset), 3001)
result = obspyNN.probability.set_probability(dataset, pdf)
obspyNN.plot.plot_stream(result[split_dataset:])
