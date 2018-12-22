import tensorflow as tf
from tensorflow import keras
from training_set import load_catalog

sfileList = ["/mnt/Data/NCU_Zland/1p/08-0831-18L.S201811"]
catalog = load_catalog(sfileList)
print(catalog)
