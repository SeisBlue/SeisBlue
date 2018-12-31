import tensorflow as tf
from tensorflow import keras
import obspyNN
from model import unet
import numpy as np

tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
                                          write_graph=True, write_images=False)
wavedir = "/mnt/Data"
sfileList = obspyNN.read_list("/mnt/tf_data/sfilelist")

dataset = obspyNN.load_dataset(sfileList, wavedir, plot=False)
wavefile, probability = obspyNN.load_training_set(dataset)

split = int(len(wavefile) * (9/10))
model = unet()
model.fit(wavefile[:split], probability[:split], epochs=300, callbacks=[tensorboard])

test_loss, test_acc = model.evaluate(wavefile, probability)
print('Test accuracy:', test_acc)

predict = model.predict(wavefile)
pdf = predict.reshape(len(dataset), 3001)
result = obspyNN.add_predict(dataset, pdf)
obspyNN.plot_stream(result[split:])
