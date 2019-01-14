from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, concatenate, Dropout, Cropping2D, ZeroPadding2D
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from tensorflow.python.keras.regularizers import l2

"""
U-net, Nest-Net model from: 
https://github.com/MrGiovanni/Nested-UNet/blob/master/model_logic.py
"""


########################################
# 2D Standard
########################################

def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):
    dropout_rate = 0.2
    act = "relu"

    x = Conv2D(nb_filter, 3, activation=act, name='conv' + stage + '_1',
               kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    x = Dropout(dropout_rate, name='dp' + stage + '_1')(x)
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv' + stage + '_2',
               kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(dropout_rate, name='dp' + stage + '_2')(x)

    return x


########################################

"""
Standard U-Net [Ronneberger et.al, 2015]
Total params: 7,759,521
"""


def U_Net(img_rows, img_cols, color_type=1, num_class=1):
    nb_filter = [8, 16, 32, 64, 128]
    # nb_filter = [32, 64, 128, 256, 512]
    pool_size = (1, 2)
    padding_size = ((0, 0), (3, 4))

    img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    zpad = ZeroPadding2D(padding_size)(img_input)

    conv1_1 = standard_unit(zpad, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D(pool_size=pool_size, name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D(pool_size=pool_size, name='pool2')(conv2_1)

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D(pool_size=pool_size, name='pool3')(conv3_1)

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D(pool_size=pool_size, name='pool4')(conv4_1)

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(nb_filter[3], 3, strides=pool_size, name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=3)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], 3, strides=pool_size, name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1], name='merge33', axis=3)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], 3, strides=pool_size, name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1], name='merge24', axis=3)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], 3, strides=pool_size, name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1], name='merge15', axis=3)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    unet_output = Conv2D(num_class, (1, 1), activation='sigmoid', name='output', kernel_initializer='he_normal',
                         padding='same', kernel_regularizer=l2(3e-4))(conv1_5)
    crop = Cropping2D(padding_size)(unet_output)

    model = Model(inputs=img_input, outputs=crop)

    return model


"""
Standard UNet++ [Zhou et.al, 2018]
Total params: 9,041,601
"""


def Nest_Net(img_rows, img_cols, color_type=1, num_class=1, deep_supervision=False):
    # nb_filter = [8, 16, 32, 64, 128]
    nb_filter = [32, 64, 128, 256, 512]
    pool_size = (1, 2)
    padding_size = ((0, 0), (3, 4))

    img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    zpad = ZeroPadding2D(padding_size)(img_input)

    conv1_1 = standard_unit(zpad, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D(pool_size=pool_size, name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D(pool_size=pool_size, name='pool2')(conv2_1)

    up1_2 = Conv2DTranspose(nb_filter[0], pool_size, strides=pool_size, name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=3)
    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D(pool_size=pool_size, name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[1], pool_size, strides=pool_size, name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=3)
    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])

    up1_3 = Conv2DTranspose(nb_filter[0], pool_size, strides=pool_size, name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=3)
    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D(pool_size=pool_size, name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[2], pool_size, strides=pool_size, name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=3)
    conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])

    up2_3 = Conv2DTranspose(nb_filter[1], pool_size, strides=pool_size, name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=3)
    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])

    up1_4 = Conv2DTranspose(nb_filter[0], pool_size, strides=pool_size, name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=3)
    conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(nb_filter[3], pool_size, strides=pool_size, name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=3)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], pool_size, strides=pool_size, name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=3)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], pool_size, strides=pool_size, name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=3)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], pool_size, strides=pool_size, name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=3)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_1', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
    nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_2', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
    nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_3', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
    nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_4', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    crop1 = Cropping2D(padding_size)(nestnet_output_1)
    crop2 = Cropping2D(padding_size)(nestnet_output_2)
    crop3 = Cropping2D(padding_size)(nestnet_output_3)
    crop4 = Cropping2D(padding_size)(nestnet_output_4)

    if deep_supervision:
        model = Model(inputs=img_input, outputs=[crop1, crop2, crop3, crop4])
    else:
        model = Model(inputs=img_input, outputs=[crop4])

    return model


"""
unet model from:
https://github.com/zhixuhao/unet/blob/master/model.py
"""


def unet(img_rows, img_cols, color_type=1, num_class=1):
    nb_filter = [8, 16, 32, 64, 128]
    # nb_filter = [32, 64, 128, 256, 512]
    padding_size = ((0, 0), (3, 4))
    pool_size = (1, 2)
    dropout_rate = 0.5

    img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    zpad = ZeroPadding2D(padding_size)(img_input)

    conv1 = Conv2D(nb_filter[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(zpad)
    conv1 = Conv2D(nb_filter[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=pool_size, padding='same')(conv1)

    conv2 = Conv2D(nb_filter[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(nb_filter[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=pool_size, padding='same')(conv2)

    conv3 = Conv2D(nb_filter[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(nb_filter[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=pool_size, padding='same')(conv3)

    conv4 = Conv2D(nb_filter[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(nb_filter[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(dropout_rate)(conv4)
    pool4 = MaxPooling2D(pool_size=pool_size, padding='same')(drop4)

    conv5 = Conv2D(nb_filter[4], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(nb_filter[4], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(dropout_rate)(conv5)

    up6 = Conv2D(nb_filter[3], 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=pool_size)(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(nb_filter[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(nb_filter[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(nb_filter[2], 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=pool_size)(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(nb_filter[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(nb_filter[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(nb_filter[1], 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=pool_size)(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(nb_filter[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(nb_filter[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(nb_filter[0], 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=pool_size)(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(nb_filter[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(nb_filter[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(num_class, 1, activation='sigmoid')(conv9)
    crop = Cropping2D(padding_size)(conv10)

    model = Model(inputs=img_input, outputs=crop)

    return model


if __name__ == '__main__':
    model = U_Net(1, 3001, 1)
    model.summary()

    model = Nest_Net(1, 3001, 1)
    model.summary()

    model = unet(1, 3001, 1)
    model.summary()
