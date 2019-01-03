from tensorflow import keras
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import multi_gpu_model

from obspy import read

import obspyNN
from obspyNN.model import unet, U_Net, Nest_Net

tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
                                          write_graph=True, write_images=False)

picked_stream = read("/mnt/tf_data/data.pkl")
wavefile, probability = obspyNN.io.load_training_set(picked_stream)

split_point = -100
X_train, X_val = wavefile[0:split_point], wavefile[split_point:]
Y_train, Y_val = probability[0:split_point], probability[split_point:]

# model = unet(1, 3001, 1)
# model = U_Net(1, 3001, 1)
model = Nest_Net(1, 3001, 1)

parallel_model = multi_gpu_model(model, gpus=2)
parallel_model.compile(optimizer=Adam(lr=3e-4), loss='binary_crossentropy', metrics=['accuracy'])
parallel_model.fit(X_train, Y_train, epochs=50, callbacks=[tensorboard])

evaluate = parallel_model.evaluate(X_val, Y_val)
print('Test loss:', evaluate[0])

parallel_model.save_weights('trained_weight.h5')

predict = parallel_model.predict(wavefile)
result = obspyNN.probability.set_probability(picked_stream, predict)
result.write("result.pkl", format="PICKLE")

