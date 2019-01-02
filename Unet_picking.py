from tensorflow import keras
from obspy import read

import obspyNN
from obspyNN.model import unet

tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
                                          write_graph=True, write_images=False)

picked_stream = read("/mnt/tf_data/dataset2017_2018HL.pkl")
wavefile, probability = obspyNN.io.load_training_set(picked_stream)

split_point = -100
X_train, X_val = wavefile[0:split_point], wavefile[split_point:]
Y_train, Y_val = probability[0:split_point], probability[split_point:]

model = unet()


while True:  # pretrain
    model.fit(X_train, Y_train, epochs=5, callbacks=[tensorboard])
    evaluate = model.evaluate(X_val, Y_val)
    print('Test loss:', evaluate[0])
    if evaluate[0] < 5:
        break
    model = unet()

while True:
    model.fit(X_train, Y_train, epochs=20, callbacks=[tensorboard])
    evaluate = model.evaluate(X_val, Y_val)
    print('Test loss:', evaluate[0])
    if evaluate[0] < 0.6:
        break

model.save_weights('trained_weight.h5')

predict = model.predict(wavefile)
result = obspyNN.probability.set_probability(picked_stream, predict)
result.write("result.pkl", format="PICKLE")

