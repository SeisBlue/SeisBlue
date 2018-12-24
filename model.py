import tensorflow as tf
from tensorflow import keras


def unet(pretrained_weights=None, input_size=(1, 7501, 1)):
    inputs = keras.layers.Input(input_size)
    zpad = keras.layers.ZeroPadding2D(((0, 0), (345, 346)))(inputs)
    conv1 = keras.layers.Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(zpad)
    conv1 = keras.layers.Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(1, 2), padding='same')(conv1)
    conv2 = keras.layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = keras.layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(1, 2), padding='same')(conv2)
    conv3 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(1, 2), padding='same')(conv3)
    conv4 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = keras.layers.Dropout(0.5)(conv4)
    pool4 = keras.layers.MaxPooling2D(pool_size=(1, 2), padding='same')(drop4)

    conv5 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = keras.layers.Dropout(0.5)(conv5)

    up6 = keras.layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        keras.layers.UpSampling2D(size=(1, 2))(drop5))
    merge6 = keras.layers.concatenate([drop4, up6], axis=3)
    conv6 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = keras.layers.Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        keras.layers.UpSampling2D(size=(1, 2))(conv6))
    merge7 = keras.layers.concatenate([conv3, up7], axis=3)
    conv7 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = keras.layers.Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        keras.layers.UpSampling2D(size=(1, 2))(conv7))
    merge8 = keras.layers.concatenate([conv2, up8], axis=3)
    conv8 = keras.layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = keras.layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = keras.layers.Conv2D(8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        keras.layers.UpSampling2D(size=(1, 2))(conv8))
    merge9 = keras.layers.concatenate([conv1, up9], axis=3)
    conv9 = keras.layers.Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = keras.layers.Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = keras.layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = keras.layers.Conv2D(1, 1, activation='sigmoid')(conv9)
    crop = keras.layers.Cropping2D(((0, 0), (345, 346)))(conv10)

    model = keras.models.Model(inputs=inputs, outputs=crop)

    model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
