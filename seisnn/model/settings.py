"""
Training step settings.
"""

import tensorflow as tf

from seisnn.model.unet import nest_net

model = nest_net(color_type=1, num_class=3)
optimizer = tf.keras.optimizers.Adam(1e-4)
bce = tf.keras.losses.BinaryCrossentropy()


@tf.function
def train_step(train_trace, train_label, val_trace, val_label):
    """
    Main training loop.

    :param train_trace: Training trace data.
    :param train_label: Training trace label.
    :param val_trace: Validation trace data.
    :param val_label: Validation trace label.
    :rtype: float
    :return: (predict loss, validation loss)
    """
    with tf.GradientTape(persistent=True) as tape:
        pred_label = model(train_trace, training=True)
        pred_loss = bce(train_label, pred_label)

        pred_val_label = model(val_trace, training=False)
        val_loss = bce(val_label, pred_val_label)

        gradients = tape.gradient(pred_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return pred_loss, val_loss


if __name__ == "__main__":
    pass
