from tensorflow.python.keras.optimizers import Adam

from seisnn.io import get_dir_list
from seisnn.pick import write_probability_pkl
from seisnn.tensorflow.generator import PredictGenerator
from seisnn.tensorflow.model import Nest_Net

pkl_dir = "/mnt/tf_data/pkl/small_set"
pkl_output_dir = pkl_dir + "_predict"
pkl_list = get_dir_list(pkl_dir)

predict_generator = PredictGenerator(pkl_list, batch_size=32)

model = Nest_Net(1, 3001, 1)
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights("/mnt/tf_data/weights/trained_weight.h5")

predict = model.predict_generator(generator=predict_generator,
                                  use_multiprocessing=True, verbose=True)

write_probability_pkl(predict, pkl_list, pkl_output_dir, remove_dir=True)
