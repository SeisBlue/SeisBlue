from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Activation, add, BatchNormalization, \
    Bidirectional, concatenate, Conv1D, Dense, Dropout, Input, Layer, \
    LayerNormalization, LSTM, MaxPooling1D, UpSampling1D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np


def transformer(img_rows=None, img_cols=None, color_type=3, num_class=3):
    inputs = Input(shape=(img_rows, img_cols, color_type))
    new_dim = tf.squeeze(inputs, axis=1, name=None)
    ff_dim = 64
    num_head = 1
    conv1 = Conv1D(8, 11, activation='relu', padding='same')(new_dim)
    pool1 = MaxPooling1D(2)(conv1)
    conv2 = Conv1D(16, 9, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling1D(2)(conv2)
    conv3 = Conv1D(16, 7, activation='relu', padding='same')(pool2)
    pool3 = MaxPooling1D(2)(conv3)
    # conv4 = Conv1D(32, 7, activation='relu', padding='same')(pool3)
    # pool4 = MaxPooling1D(2)(conv4)
    conv5 = Conv1D(32, 5, activation='relu', padding='same')(pool3)
    pool5 = MaxPooling1D(2)(conv5)
    conv6 = Conv1D(64, 5, activation='relu', padding='same')(pool5)
    pool6 = MaxPooling1D(2)(conv6)
    conv7 = Conv1D(64, 3, activation='relu', padding='same')(pool6)
    pool7 = MaxPooling1D(2)(conv7)
    resCNN = ResNet_build(64, 3)
    res1 = resCNN(pool7)

    bilstm1 = Bidirectional(LSTM(64, return_sequences=True))(res1)
    bilstm1 = Dropout(0.1)(bilstm1)
    bilstm1 = Conv1D(64, 1, activation='relu', padding='same')(bilstm1)
    bilstm1 = LayerNormalization(epsilon=1e-6)(bilstm1)
    bilstm1 = Activation('relu')(bilstm1)

    bilstm2 = Bidirectional(LSTM(64, return_sequences=True))(bilstm1)
    bilstm2 = Dropout(0.1)(bilstm2)
    bilstm2 = Conv1D(64, 1, activation='relu', padding='same')(bilstm2)
    bilstm2 = LayerNormalization(epsilon=1e-6)(bilstm2)
    bilstm2 = Activation('relu')(bilstm2)

    lstm1 = LSTM(64, return_sequences=True)(bilstm2)
    transformer_block1 = TransformerBlockE(64, num_head, ff_dim)
    transE = transformer_block1(lstm1)
    transformer_block2 = TransformerBlockE(64, num_head, ff_dim)
    transE = transformer_block2(transE)
    up1 = UpSampling1D(size=2)(transE)
    conv8 = Conv1D(96, 3, activation='relu', padding='same')(up1)
    up2 = UpSampling1D(size=2)(conv8)
    conv9 = Conv1D(96, 5, activation='relu', padding='same')(up2)
    up3 = UpSampling1D(size=2)(conv9)
    conv10 = Conv1D(32, 5, activation='relu', padding='same')(up3)
    # up4 = UpSampling1D(size=2)(conv10)
    # conv11 = Conv1D(32, 7, activation='relu', padding='same')(up4)
    up5 = UpSampling1D(size=2)(conv10)
    conv12 = Conv1D(16, 7, activation='relu', padding='same')(up5)
    up6 = UpSampling1D(size=2)(conv12)
    conv13 = Conv1D(16, 9, activation='relu', padding='same')(up6)
    up7 = UpSampling1D(size=2)(conv13)
    conv14 = Conv1D(8, 11, activation='relu', padding='same')(up7)

    conv15 = Conv1D(1, 1, activation='sigmoid', padding='same')(conv14)
    #############################################################################
    lstm2 = LSTM(64, return_sequences=True)(transE)
    transformer_block3 = TransformerBlockE(64, num_head, ff_dim)
    transE_P = transformer_block3(lstm2)
    up1_P = UpSampling1D(size=2)(transE_P)
    conv8_P = Conv1D(96, 3, activation='relu', padding='same')(up1_P)
    up2_P = UpSampling1D(size=2)(conv8_P)
    conv9_P = Conv1D(96, 5, activation='relu', padding='same')(up2_P)
    up3_P = UpSampling1D(size=2)(conv9_P)
    conv10_P = Conv1D(32, 5, activation='relu', padding='same')(up3_P)
    # up4_P = UpSampling1D(size=2)(conv10_P)
    # conv11_P = Conv1D(32, 7, activation='relu', padding='same')(up4_P)
    up5_P = UpSampling1D(size=2)(conv10_P)
    conv12_P = Conv1D(16, 7, activation='relu', padding='same')(up5_P)
    up6_P = UpSampling1D(size=2)(conv12_P)
    conv13_P = Conv1D(16, 9, activation='relu', padding='same')(up6_P)
    up7_P = UpSampling1D(size=2)(conv13_P)
    conv14_P = Conv1D(8, 11, activation='relu', padding='same')(up7_P)
    conv15_P = Conv1D(1, 1, activation='sigmoid', padding='same')(conv14_P)
    #############################################################################
    lstm3 = LSTM(64, return_sequences=True)(transE)
    transformer_block4 = TransformerBlockE(64, num_head, ff_dim)
    transE_S = transformer_block4(lstm3)
    up1_S = UpSampling1D(size=2)(transE_S)
    conv8_S = Conv1D(96, 3, activation='relu', padding='same')(up1_S)
    up2_S = UpSampling1D(size=2)(conv8_S)
    conv9_S = Conv1D(96, 5, activation='relu', padding='same')(up2_S)
    up3_S = UpSampling1D(size=2)(conv9_S)
    conv10_S = Conv1D(32, 5, activation='relu', padding='same')(up3_S)
    # up4_S = UpSampling1D(size=2)(conv10_S)
    # conv11_S = Conv1D(32, 7, activation='relu', padding='same')(up4_S)
    up5_S = UpSampling1D(size=2)(conv10_S)
    conv12_S = Conv1D(16, 7, activation='relu', padding='same')(up5_S)
    up6_S = UpSampling1D(size=2)(conv12_S)
    conv13_S = Conv1D(16, 9, activation='relu', padding='same')(up6_S)
    up7_S = UpSampling1D(size=2)(conv13_S)
    conv14_S = Conv1D(8, 11, activation='relu', padding='same')(up7_S)
    conv15_S = Conv1D(1, 1, activation='sigmoid', padding='same')(conv14_S)
    #############################################################################
    ouput = concatenate([conv15, conv15_P, conv15_S], axis=2)
    ouput = tf.expand_dims(ouput, 1, name=None)
    model = Model(inputs=inputs, outputs=ouput)

    opt = Adam(lr=1e-4)
    model.compile(optimizer=opt, loss="binary_crossentropy",
                  metrics=['accuracy'])
    model.summary()

    return model


class ResBlock(Layer):
    def __init__(self, filter_nums, strides=1, residual_path=False, **kwargs):
        super(ResBlock, self).__init__()
        self.filter_nums = filter_nums
        self.strides = strides
        self.residual_path = residual_path
        self.bn_1 = BatchNormalization()
        self.act_relu1 = Activation('relu')
        self.drop_1 = Dropout(0.1)
        self.conv_1 = Conv1D(filter_nums, 3, strides=strides,
                             padding='same')
        self.bn_2 = BatchNormalization()
        self.act_relu2 = Activation('relu')
        self.drop_2 = Dropout(0.1)
        self.conv_2 = Conv1D(filter_nums, 3, strides=1,
                             padding='same')

        if strides != 1:
            self.block = Sequential()
            self.block.add(Conv1D(filter_nums, 1, strides=strides))
        else:
            self.block = lambda x: x

    def call(self, inputs, training=None):

        x = self.bn_1(inputs, training=training)
        x = self.act_relu1(x)
        x = self.drop_1(x)
        x = self.conv_1(x)
        x = self.bn_2(x, training=training)
        x = self.act_relu2(x)
        x = self.drop_2(x)
        x = self.conv_2(x)

        identity = self.block(inputs)
        outputs = add([x, identity])
        outputs = tf.nn.relu(outputs)

        return outputs

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filter_nums': self.filter_nums,
            'strides': self.strides,
            'residual_path': self.residual_path,
        })
        return config


def ResNet_build(filter_nums, block_nums, strides=1):
    build_model = Sequential()
    build_model.add(ResBlock(filter_nums, strides))
    for _ in range(1, block_nums):
        build_model.add(ResBlock(filter_nums, strides=1))
    return build_model


class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads=4, mask=False, **kwargs):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)
        self.mask = mask

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)

        if self.mask:
            infmatrix = np.ones([self.seq_dim, self.seq_dim]) * -np.inf
            infmatrix = np.triu(infmatrix, 1)
            infmatrix = tf.constant(infmatrix, tf.float32)
            scaled_score = scaled_score + infmatrix
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x,
                       (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        self.seq_dim = tf.shape(inputs)[1]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'mask': self.mask,
        })
        return config


class TransformerBlockE(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlockE, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.att1 = MultiHeadSelfAttention(embed_dim, num_heads, mask=False)

        self.ffn1 = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"),
             Dense(embed_dim), ]
        )

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, **kwrags):
        att1_output = self.att1(inputs)
        att1_output = self.dropout1(att1_output, training=True)
        out1 = self.layernorm1(inputs + att1_output)
        ffn_output = self.ffn1(out1)
        ffn_output = self.dropout2(ffn_output, training=True)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config().copy()
        config.update({

            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
        })
        return config


class TransformerBlockD(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlockD, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.att1 = MultiHeadSelfAttention(embed_dim, num_heads, mask=True)
        self.att2 = MultiHeadSelfAttention(embed_dim, num_heads, mask=False)
        self.ffn1 = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"),
             Dense(embed_dim), ]
        )

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

    def call(self, inputs, **kwargs):
        att1_output = self.att1(inputs[1])
        att1_output = self.dropout1(att1_output, training=True)
        out1 = self.layernorm1(inputs[1] + att1_output)
        att2_output = self.att2(inputs[0] + out1)
        att2_output = self.dropout2(att2_output, training=True)
        out2 = self.layernorm2(att2_output + out1)
        ffn_output = self.ffn1(out2)
        ffn_output = self.dropout3(ffn_output, training=True)
        out3 = self.layernorm3(out2 + ffn_output)
        return out3

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
        })
        return config
