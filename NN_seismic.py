import tensorflow as tf
from tensorflow import keras
import obspyNN

sfileList = [["/mnt/Data/NCU_Zland/1p/08-0831-18L.S201811", "/mnt/Data/NCU_Zland/1p/"]]
dataset = obspyNN.load_dataset(sfileList, plot=True)
wavefile, probability = obspyNN.load_training_set(dataset)
