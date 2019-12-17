import tensorflow as tf
from seisnn.model.unet import Nest_Net

model = Nest_Net()
optimizer = tf.keras.optimizers.Adam(3e-4)
bce = tf.keras.losses.BinaryCrossentropy()


@tf.function
def train_step(trace, real_pdf):
    with tf.GradientTape(persistent=True) as tape:
        pred_pdf = model(trace, training=True)
        pred_loss = bce(real_pdf, pred_pdf)
        gradients = tape.gradient(pred_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return pred_loss