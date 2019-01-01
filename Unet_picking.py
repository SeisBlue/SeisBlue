from tensorflow import keras
from obspy import read

import obspyNN
from obspyNN.model import unet

tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
                                          write_graph=True, write_images=False)

picked_stream = read("/mnt/tf_data/dataset.pkl")
wavefile, probability = obspyNN.io.load_training_set(picked_stream)

split_point = -20
X_train, X_val = wavefile[0:split_point], wavefile[split_point:]
Y_train, Y_val = probability[0:split_point], probability[split_point:]

model = unet()
model.fit(X_train, Y_train, epochs=5, callbacks=[tensorboard])

test_loss, test_acc = model.evaluate(X_val, Y_val)
print('Test accuracy:', test_acc)

predict = model.predict(wavefile)
result = obspyNN.probability.set_probability(picked_stream, predict)
obspyNN.plot.plot_stream(result[split_point:])
