from tensorflow import keras
from obspy import read

import obspyNN
from obspyNN.model import unet

tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
                                          write_graph=True, write_images=False)

picked_stream = read("/mnt/tf_data/dataset.pkl")
wavefile, probability = obspyNN.io.load_training_set(picked_stream)

split_dataset = -20

training_data = wavefile[:split_dataset]
training_label = probability[:split_dataset]

test_data = wavefile[split_dataset:]
test_label = probability[split_dataset:]

model = unet()
model.fit(training_data, training_label, epochs=5, callbacks=[tensorboard])

test_loss, test_acc = model.evaluate(test_data, test_label)
print('Test accuracy:', test_acc)

predict = model.predict(wavefile)
result = obspyNN.probability.set_probability(picked_stream, predict)
obspyNN.plot.plot_stream(result[split_dataset:])
