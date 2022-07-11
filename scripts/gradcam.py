import seisblue
import tensorflow as tf
from seisblue.model.attention import TransformerBlockE, TransformerBlockD, \
    MultiHeadSelfAttention, ResBlock
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def get_saliency_map(model, image, y):
    fig, axes = plt.subplots(3, 1, figsize=(8, 2))
    for i in range(3):
        with tf.GradientTape() as tape:
            tape.watch(image)
            predictions = model(image)
            loss = tf.keras.losses.BinaryCrossentropy()
            loss = loss(y[:, :, :, i], predictions[:, :, :, i])
        gradient = tape.gradient(loss, image)

        # take maximum across channels
        gradient = tf.reduce_max(gradient, axis=-1)

        # convert to numpy
        gradient = gradient.numpy()

        # normaliz between 0 and 1
        min_val, max_val = np.min(gradient), np.max(gradient)
        smap = (gradient - min_val) / (
                    max_val - min_val + tf.keras.backend.epsilon())
        sns.heatmap(smap[0], vmin=0, vmax=1, ax=axes[i], cbar=False)
    plt.show()
    plt.close()


model = tf.keras.models.load_model(
    '/home/andy/Models/HP2017_EQ_NEW_1_45.h5',
    custom_objects={
        'TransformerBlockE': TransformerBlockE,
        'TransformerBlockD': TransformerBlockD,
        'MultiHeadSelfAttention': MultiHeadSelfAttention,
        'ResBlock': ResBlock
    })
submodel = tf.keras.models.Model(inputs=model.input,
                                 outputs=model.layers[-1].output)
dataset = seisblue.io.read_dataset([
    '/home/andy/TFRecord/noise/noise_EQ_NEW_1_45.tfrecord'
])
for item in dataset:
    instance = seisblue.core.Instance(item)
    instance.plot(threshold=0.4)
    x = tf.reshape(instance.trace.data, (1, 1, 3008, 3))
    y = tf.reshape(instance.label.data, (1, 1, 3008, 3))
    get_saliency_map(submodel, x, y)
