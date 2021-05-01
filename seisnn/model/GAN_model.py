import tensorflow as tf
from tensorflow.keras.layers import concatenate, Conv2D, Conv2DTranspose, \
    Dropout, Input, MaxPooling2D, LeakyReLU, BatchNormalization, Flatten, \
    Dense, Embedding, Reshape, Concatenate
from tensorflow.keras.regularizers import l2


def _standard_unit(input_tensor, stage, nb_filter, kernel_size):
    """
    2D standard unit.

    :param input_tensor:
    :param str stage: Stage name.
    :param int nb_filter: Filter number.
    :param tuple kernel_size: Kernel size.
    :return: tensor
    """
    dropout_rate = 0.1
    act = "relu"

    x = Conv2D(nb_filter, kernel_size,
               activation=act, padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4),
               name='conv' + stage + '_1')(input_tensor)
    x = Dropout(dropout_rate, name='dp' + stage + '_1')(x)
    x = Conv2D(nb_filter, kernel_size,
               activation=act, padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4),
               name='conv' + stage + '_2')(x)
    x = Dropout(dropout_rate, name='dp' + stage + '_2')(x)

    return x


def build_discriminator(img_rows=1, img_cols=3008, color_type=3,
                        num_class=3):
    img_input = Input(shape=(img_rows, img_cols, color_type * 2),
                      name='main_input')
    conv1 = Conv2D(64, (1, 11), activation='relu', padding='same')(img_input)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    conv2 = Conv2D(64, (1, 11), activation='relu', padding='same')(conv1)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = (LeakyReLU(alpha=0.1))(conv2)
    conv3 = Conv2D(128, (1, 5), activation='relu', padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = (LeakyReLU(alpha=0.1))(conv3)
    flat1 = Flatten()(conv3)
    output = Dense(1, activation='sigmoid')(flat1)
    model = tf.keras.Model(inputs=img_input, outputs=output)
    model.summary()
    return model

def build_patch_discriminator(img_rows=1, img_cols=3008, color_type=3,
                        num_class=3):
    img_input = Input(shape=(img_rows, img_cols, color_type * 2),
                      name='main_input')
    conv1 = Conv2D(64, (1, 32),strides=4, activation='relu', padding='same')(img_input)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    conv2 = Conv2D(128, (1, 16),strides=4, activation='relu', padding='same')(conv1)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = (LeakyReLU(alpha=0.1))(conv2)
    conv3 = Conv2D(256, (1, 8),strides=4, activation='relu')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = (LeakyReLU(alpha=0.1))(conv3)
    output = Conv2D(1,(1,4),strides=4,activation='sigmoid')(conv3)

    model = tf.keras.Model(inputs=img_input, outputs=output)
    model.summary()
    return model


def build_cgan(generator, discriminator):
    """
    :param generator:
    :param discriminator:
    :return:

    """
    trace = Input(shape=(1, 3008, 3))
    gen_label = generator(trace)
    concat = concatenate([trace, gen_label], axis=3)
    classification = discriminator(concat)
    model = tf.keras.Model([trace],
                           [gen_label, classification])
    return model
