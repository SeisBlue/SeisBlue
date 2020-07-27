"""
Training step settings.
"""

import tensorflow as tf

from seisnn.model.unet import nest_net

model = nest_net()
optimizer = tf.keras.optimizers.Adam(1e-4)
bce = tf.keras.losses.BinaryCrossentropy()


@tf.function
def train_step(train_trace, train_pdf, val_trace, val_pdf):
    """
    Main training loop.

    :param train_trace:
    :param train_pdf:
    :param val_trace:
    :param val_pdf:
    :return:
    """
    with tf.GradientTape(persistent=True) as tape:
        pred_pdf = model(train_trace, training=True)
        pred_loss = bce(train_pdf, pred_pdf)

        pred_val_pdf = model(val_trace, training=False)
        val_loss = bce(val_pdf, pred_val_pdf)

        gradients = tape.gradient(pred_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return pred_loss, val_loss


if __name__ == "__main__":
    pass
