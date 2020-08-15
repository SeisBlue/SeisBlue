"""
Unet model structure.

unet, nest_net model modified from:
https://github.com/MrGiovanni/UNetPlusPlus/blob/master/helper_functions.py
"""
import tensorflow as tf
from tensorflow.keras.layers import concatenate, Conv2D, Conv2DTranspose, \
    Dropout, Input, MaxPooling2D
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


def unet(img_rows, img_cols, color_type=1, num_class=1):
    """
    Standard U-Net [Ronneberger et.al, 2015]

    Total params: 420,249

    :param int img_rows: Height of the data.
    :param int img_cols: Width of the data.
    :param int color_type: Channel number of the data.
    :param int num_class: Output class number.
    :return: UNet model.
    """

    nb_filter = [8, 11, 16, 22, 32]
    # nb_filter = [8, 16, 32, 64, 128]
    pool_size = (1, 2)
    kernel_size = (1, 7)

    img_input = Input(shape=(img_rows, img_cols, color_type),
                      name='main_input')

    conv1_1 = _standard_unit(img_input, stage='11', nb_filter=nb_filter[0],
                             kernel_size=kernel_size)
    pool1 = MaxPooling2D(pool_size=pool_size, name='pool1')(conv1_1)

    conv2_1 = _standard_unit(pool1, stage='21', nb_filter=nb_filter[1],
                             kernel_size=kernel_size)
    pool2 = MaxPooling2D(pool_size=pool_size, name='pool2')(conv2_1)

    conv3_1 = _standard_unit(pool2, stage='31', nb_filter=nb_filter[2],
                             kernel_size=kernel_size)
    pool3 = MaxPooling2D(pool_size=pool_size, name='pool3')(conv3_1)

    conv4_1 = _standard_unit(pool3, stage='41', nb_filter=nb_filter[3],
                             kernel_size=kernel_size)
    pool4 = MaxPooling2D(pool_size=pool_size, name='pool4')(conv4_1)

    conv5_1 = _standard_unit(pool4, stage='51', nb_filter=nb_filter[4],
                             kernel_size=kernel_size)

    up4_2 = Conv2DTranspose(nb_filter[3], kernel_size, strides=pool_size,
                            padding='same', name='up42')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], axis=3, name='merge42')
    conv4_2 = _standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3],
                             kernel_size=kernel_size)

    up3_3 = Conv2DTranspose(nb_filter[2], kernel_size, strides=pool_size,
                            padding='same', name='up33')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1], axis=3, name='merge33')
    conv3_3 = _standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2],
                             kernel_size=kernel_size)

    up2_4 = Conv2DTranspose(nb_filter[1], kernel_size, strides=pool_size,
                            padding='same', name='up24')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1], axis=3, name='merge24')
    conv2_4 = _standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1],
                             kernel_size=kernel_size)

    up1_5 = Conv2DTranspose(nb_filter[0], kernel_size, strides=pool_size,
                            padding='same', name='up15')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1], axis=3, name='merge15')
    conv1_5 = _standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0],
                             kernel_size=kernel_size)

    unet_output = Conv2D(num_class, (1, 1),
                         activation='sigmoid', padding='same',
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2(3e-4),
                         name='output')(conv1_5)

    model = tf.keras.Model(inputs=img_input, outputs=unet_output)

    return model


def nest_net(img_rows=None, img_cols=None, color_type=1, num_class=1):
    """
    Standard UNet++ [Zhou et.al, 2018]

    Total params: 496,225

    :param int img_rows: Height of the data.
    :param int img_cols: Width of the data.
    :param int color_type: Channel number of the data.
    :param int num_class: Output class number.
    :return: Nest net model.
    """
    nb_filter = [8, 16, 32, 64, 128]
    pool_size = (1, 2)
    kernel_size = (1, 7)

    img_input = Input(shape=(img_rows, img_cols, color_type),
                      name='main_input')

    conv1_1 = _standard_unit(img_input, stage='11', nb_filter=nb_filter[0],
                             kernel_size=kernel_size)
    pool1 = MaxPooling2D(pool_size=pool_size, name='pool1')(conv1_1)

    conv2_1 = _standard_unit(pool1, stage='21', nb_filter=nb_filter[1],
                             kernel_size=kernel_size)
    pool2 = MaxPooling2D(pool_size=pool_size, name='pool2')(conv2_1)

    up1_2 = Conv2DTranspose(nb_filter[0], kernel_size, strides=pool_size,
                            padding='same', name='up12')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], axis=3, name='merge12')
    conv1_2 = _standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0],
                             kernel_size=kernel_size)

    conv3_1 = _standard_unit(pool2, stage='31', nb_filter=nb_filter[2],
                             kernel_size=kernel_size)
    pool3 = MaxPooling2D(pool_size=pool_size, name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[1], kernel_size, strides=pool_size,
                            padding='same', name='up22')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], axis=3, name='merge22')
    conv2_2 = _standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1],
                             kernel_size=kernel_size)

    up1_3 = Conv2DTranspose(nb_filter[0], kernel_size, strides=pool_size,
                            padding='same', name='up13')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], axis=3, name='merge13')
    conv1_3 = _standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0],
                             kernel_size=kernel_size)

    conv4_1 = _standard_unit(pool3, stage='41', nb_filter=nb_filter[3],
                             kernel_size=kernel_size)
    pool4 = MaxPooling2D(pool_size=pool_size, name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[2], kernel_size, strides=pool_size,
                            padding='same', name='up32')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], axis=3, name='merge32')
    conv3_2 = _standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2],
                             kernel_size=kernel_size)

    up2_3 = Conv2DTranspose(nb_filter[1], kernel_size, strides=pool_size,
                            padding='same', name='up23')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], axis=3, name='merge23')
    conv2_3 = _standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1],
                             kernel_size=kernel_size)

    up1_4 = Conv2DTranspose(nb_filter[0], kernel_size, strides=pool_size,
                            padding='same', name='up14')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], axis=3,
                          name='merge14')
    conv1_4 = _standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0],
                             kernel_size=kernel_size)

    conv5_1 = _standard_unit(pool4, stage='51', nb_filter=nb_filter[4],
                             kernel_size=kernel_size)

    up4_2 = Conv2DTranspose(nb_filter[3], kernel_size, strides=pool_size,
                            padding='same', name='up42')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], axis=3, name='merge42')
    conv4_2 = _standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3],
                             kernel_size=kernel_size)

    up3_3 = Conv2DTranspose(nb_filter[2], kernel_size, strides=pool_size,
                            padding='same', name='up33')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], axis=3, name='merge33')
    conv3_3 = _standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2],
                             kernel_size=kernel_size)

    up2_4 = Conv2DTranspose(nb_filter[1], kernel_size, strides=pool_size,
                            padding='same', name='up24')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], axis=3,
                          name='merge24')
    conv2_4 = _standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1],
                             kernel_size=kernel_size)

    up1_5 = Conv2DTranspose(nb_filter[0], kernel_size, strides=pool_size,
                            padding='same', name='up15')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], axis=3,
                          name='merge15')
    conv1_5 = _standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0],
                             kernel_size=kernel_size)

    nestnet_output = Conv2D(num_class, (1, 1),
                            activation='sigmoid', padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4),
                            name='output')(conv1_5)

    model = tf.keras.Model(inputs=img_input, outputs=[nestnet_output])

    return model


if __name__ == '__main__':
    # model = U_Net(1, 3008, 1)
    # model.summary()

    model = nest_net(1, 3008, 1, num_class=3)
    model.summary()
