"""
TEAM model modified from:
https://github.com/yetinam/TEAM
"""
import tensorflow as tf
from tensorflow.keras import initializers, backend as K, \
    Input, Model, Sequential
from tensorflow.keras.layers import Add, Concatenate, Conv1D, Conv2D, \
    Dense, Dropout, Embedding, Flatten, GlobalMaxPooling1D, \
    Lambda, Layer, Masking, MaxPooling1D, Reshape, TimeDistributed
import numpy as np
import os
import pickle


class MLP(Sequential):
    def __init__(self, input_shape, dims=(100, 50), activation='relu', last_activation=None):
        Sequential.__init__(self)
        if last_activation is None:
            last_activation = activation
        self.add(Dense(dims[0], activation=activation, input_shape=input_shape))
        for d in dims[1:-1]:
            self.add(Dense(d, activation=activation))
        self.add(Dense(dims[-1], activation=last_activation))


class MixtureOutput(Model):
    def __init__(self, input_shape, n, d=1, activation='relu', eps=1e-4, bias_mu=1.8, bias_sigma=0.2,
                 name=None):
        inp_masked = Input(shape=input_shape)
        inp = StripMask()(inp_masked)

        alpha = Dense(n, activation='softmax')(inp)
        alpha = Reshape((n, 1))(alpha)

        mu = Dense(n * d, activation=activation, bias_initializer=initializers.Constant(bias_mu))(inp)
        mu = Reshape((n, d))(mu)

        sigma = Dense(n * d, activation='relu', bias_initializer=initializers.Constant(bias_sigma))(inp)
        sigma = Lambda(lambda x: x + eps)(sigma)  # Add epsilon to avoid division by 0
        sigma = Reshape((n, d))(sigma)

        out = Concatenate(axis=2)([alpha, mu, sigma])

        Model.__init__(self, inputs=inp_masked, outputs=out, name=name)


class NormalizedScaleEmbedding(Model):
    def __init__(self, input_shape, activation='relu', downsample=1, mlp_dims=(500, 300, 200, 150), eps=1e-8):
        self.activation = activation
        self.inp_shape = input_shape
        self.downsample = downsample
        self.mlp_dims = mlp_dims
        self.eps = eps
        inp, out = self._build_model()
        Model.__init__(self, inputs=inp, outputs=out)

    def _build_model(self):
        activation = self.activation
        downsample = self.downsample
        inp = Input(shape=self.inp_shape)
        x = Lambda(lambda t: t / (K.max(K.abs(t), axis=(1, 2), keepdims=True) + self.eps))(inp)
        x = Lambda(lambda t: K.expand_dims(t))(x)
        scale = Lambda(lambda t: K.log(K.max(K.abs(t), axis=(1, 2)) + self.eps) / 100)(inp)
        scale = Lambda(lambda t: K.expand_dims(t))(scale)
        x = Conv2D(8, (downsample, 1), strides=(downsample, 1), activation=activation)(x)
        x = Conv2D(32, (16, 3), strides=(1, 3), activation=activation)(x)
        x = Reshape((-1, 32 * self.inp_shape[-1] // 3))(x)
        x = Conv1D(64, 16, activation=activation)(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(128, 16, activation=activation)(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(32, 8, activation=activation)(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(32, 8, activation=activation)(x)
        x = Conv1D(16, 4, activation=activation)(x)

        x = Flatten()(x)

        x = Concatenate()([x, scale])

        x = MLP(input_shape=(865,), dims=self.mlp_dims, activation=activation)(x)

        return inp, x


class Transformer(Model):
    def __init__(self, max_stations=32, emb_dim=500, layers=6, att_masking=False, hidden_dropout=0.0,
                 mad_params={}, ffn_params={}, norm_params={}):
        self.blocks = [(MultiHeadSelfAttention(**mad_params),
                        PointwiseFeedForward(**ffn_params),
                        LayerNormalization(**norm_params),
                        LayerNormalization(**norm_params))
                       for _ in range(layers)]
        inp = Input((max_stations, emb_dim))
        if att_masking:
            att_mask = Input((max_stations,), dtype=bool)
        else:
            att_mask = None
        x = inp
        for attention_layer, ffn_layer, norm1_layer, norm2_layer in self.blocks:
            if att_mask is not None:
                modified_x = attention_layer([x, att_mask])
            else:
                modified_x = attention_layer(x)
            if hidden_dropout > 0:
                modified_x = Dropout(hidden_dropout)(modified_x)
            x = norm1_layer(Add()([x, modified_x]))
            modified_x = ffn_layer(x)
            if hidden_dropout > 0:
                modified_x = Dropout(hidden_dropout)(modified_x)
            x = norm2_layer(Add()([x, modified_x]))
        inputs = inp
        if att_masking:
            inputs = [inp, att_mask]
        super(Transformer, self).__init__(inputs=inputs, outputs=x)


# Calculates and concatenates sinusoidal embeddings for lat, lon and depth
# Note: Permutation is completely unnecessary, but kept for compatibility reasons
# WARNING: Does not take into account curvature of the earth!
class PositionEmbedding(Layer):
    def __init__(self, wavelengths, emb_dim, borehole=False, rotation=None, rotation_anchor=None, **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.wavelengths = wavelengths  # Format: [(min_lat, max_lat), (min_lon, max_lon), (min_depth, max_depth)]
        self.emb_dim = emb_dim
        self.borehole = borehole
        self.rotation = rotation
        self.rotation_anchor = rotation_anchor

        if rotation is not None and rotation_anchor is None:
            raise ValueError('Rotations in the positional embedding require a rotation anchor')

        if rotation is not None:
            # print(f'Rotating by {np.rad2deg(rotation)} degrees')
            c, s = np.cos(rotation), np.sin(rotation)
            self.rotation_matrix = K.variable(np.array(((c, -s), (s, c))), dtype=K.floatx())
        else:
            self.rotation_matrix = None

        min_lat, max_lat = wavelengths[0]
        min_lon, max_lon = wavelengths[1]
        min_depth, max_depth = wavelengths[2]
        assert emb_dim % 10 == 0
        if borehole:
            assert emb_dim % 20 == 0
        lat_dim = emb_dim // 5
        lon_dim = emb_dim // 5
        depth_dim = emb_dim // 10
        if borehole:
            depth_dim = emb_dim // 20
        self.lat_coeff = 2 * np.pi * 1. / min_lat * ((min_lat / max_lat) ** (np.arange(lat_dim) / lat_dim))
        self.lon_coeff = 2 * np.pi * 1. / min_lon * ((min_lon / max_lon) ** (np.arange(lon_dim) / lon_dim))
        self.depth_coeff = 2 * np.pi * 1. / min_depth * ((min_depth / max_depth) ** (np.arange(depth_dim) / depth_dim))
        lat_sin_mask = np.arange(emb_dim) % 5 == 0
        lat_cos_mask = np.arange(emb_dim) % 5 == 1
        lon_sin_mask = np.arange(emb_dim) % 5 == 2
        lon_cos_mask = np.arange(emb_dim) % 5 == 3
        depth_sin_mask = np.arange(emb_dim) % 10 == 4
        depth_cos_mask = np.arange(emb_dim) % 10 == 9
        self.mask = np.zeros(emb_dim)
        self.mask[lat_sin_mask] = np.arange(lat_dim)
        self.mask[lat_cos_mask] = lat_dim + np.arange(lat_dim)
        self.mask[lon_sin_mask] = 2 * lat_dim + np.arange(lon_dim)
        self.mask[lon_cos_mask] = 2 * lat_dim + lon_dim + np.arange(lon_dim)
        if borehole:
            depth_dim *= 2
        self.mask[depth_sin_mask] = 2 * lat_dim + 2 * lon_dim + np.arange(depth_dim)
        self.mask[depth_cos_mask] = 2 * lat_dim + 2 * lon_dim + depth_dim + np.arange(depth_dim)
        self.mask = self.mask.astype('int32')
        self.fake_borehole = False

    def build(self, input_shape):
        if input_shape[-1] == 3:
            self.fake_borehole = True
        super(PositionEmbedding, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, mask=None):
        if self.rotation is not None:
            lat_base = x[:, :, 0]
            lon_base = x[:, :, 1]
            lon_base *= K.cos(lat_base * np.pi / 180)

            lat_base -= self.rotation_anchor[0]
            lon_base -= self.rotation_anchor[1] * K.cos(self.rotation_anchor[0] * np.pi / 180)

            latlon = K.stack([lat_base, lon_base], axis=-1)
            rotated = latlon @ self.rotation_matrix

            lat_base = rotated[:, :, 0:1] * self.lat_coeff
            lon_base = rotated[:, :, 1:2] * self.lon_coeff
            depth_base = x[:, :, 2:3] * self.depth_coeff
        else:
            lat_base = x[:, :, 0:1] * self.lat_coeff
            lon_base = x[:, :, 1:2] * self.lon_coeff
            depth_base = x[:, :, 2:3] * self.depth_coeff
        if self.borehole:
            if self.fake_borehole:
                # Use third value for the depth of the top station and 0 for the borehole depth
                depth_base = x[:, :, 2:3] * self.depth_coeff * 0
                depth2_base = x[:, :, 2:3] * self.depth_coeff
            else:
                depth2_base = x[:, :, 3:4] * self.depth_coeff
            output = tf.concat([K.sin(lat_base), K.cos(lat_base),
                                K.sin(lon_base), K.cos(lon_base),
                                K.sin(depth_base), K.cos(depth_base),
                                K.sin(depth2_base), K.cos(depth2_base)], axis=-1)
        else:
            output = tf.concat([K.sin(lat_base), K.cos(lat_base),
                                K.sin(lon_base), K.cos(lon_base),
                                K.sin(depth_base), K.cos(depth_base)], axis=-1)
        output = tf.gather(output, self.mask, axis=-1)
        if mask is not None:
            mask = K.expand_dims(K.cast(mask, K.floatx()), axis=-1)
            output *= mask  # Zero out all masked elements
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.emb_dim,)

    def compute_mask(self, inputs, mask=None):
        return mask


class MultiHeadSelfAttention(Layer):
    def __init__(self, n_heads, infinity=1e6,
                 att_masking=False,
                 kernel_initializer=initializers.RandomUniform(minval=-1.2, maxval=1.2),
                 att_dropout=0.0,
                 **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.n_heads = n_heads
        self.infinity = infinity
        # Attention masking: Model may only attend to stations where attention mask is true
        # Different from regular masking, as masked (i.e. att_mask = False) stations still collect information
        self.att_masking = att_masking
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.att_dropout = att_dropout

    def build(self, input_shape):
        if self.att_masking:
            input_shape = input_shape[0]
        n_heads = self.n_heads
        d_model = input_shape[-1]  # Embedding dim
        self.stations = input_shape[1]
        assert d_model % n_heads == 0
        d_key = d_model // n_heads  # = d_query = d_val
        self.d_key = d_key
        self.WQ = self.add_weight('WQ', (d_model, d_key * n_heads), initializer=self.kernel_initializer)
        self.WK = self.add_weight('WK', (d_model, d_key * n_heads), initializer=self.kernel_initializer)
        self.WV = self.add_weight('WV', (d_model, d_key * n_heads), initializer=self.kernel_initializer)
        self.WO = self.add_weight('WO', (d_key * n_heads, d_model), initializer=self.kernel_initializer)
        super(MultiHeadSelfAttention, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, mask=None):
        d_key = self.d_key
        n_heads = self.n_heads
        if self.att_masking:
            att_mask = x[1]
            x = x[0]
            if mask is not None:
                mask = mask[0]
        else:
            att_mask = None
        q = K.dot(x, self.WQ)  # (batch, stations, key*n_heads)
        q = K.reshape(q, (-1, self.stations, d_key, n_heads))
        q = K.permute_dimensions(q, [0, 3, 1, 2])  # (batch, n_heads, stations, key)
        k = K.dot(x, self.WK)  # (batch, stations, key*n_heads)
        k = K.reshape(k, (-1, self.stations, d_key, n_heads))
        k = K.permute_dimensions(k, [0, 3, 2, 1])  # (batch, n_heads, key, stations)
        score = tf.matmul(q, k) / np.sqrt(d_key)  # (batch, n_heads, stations, stations)
        if mask is not None:
            inv_mask = K.expand_dims(K.expand_dims(K.cast(~mask, K.floatx()), axis=-1), axis=-1)  # (batch, stations, 1, 1)
            mask_B = K.permute_dimensions(inv_mask, [0, 2, 3, 1])  # (batch, 1, 1, stations)
            score = score - mask_B * self.infinity
        if att_mask is not None:
            inv_mask = K.expand_dims(K.expand_dims(K.cast(~att_mask, K.floatx()), axis=-1),
                                     axis=-1)  # (batch, stations, 1, 1)
            mask_B = K.permute_dimensions(inv_mask, [0, 2, 3, 1])  # (batch, 1, 1, stations)
            score = score - mask_B * self.infinity
        score = K.softmax(score)
        if self.att_dropout > 0:
            score = K.dropout(score, self.att_dropout)
        v = K.dot(x, self.WV)  # (batch, stations, key*n_heads)
        v = K.reshape(v, (-1, self.stations, d_key, n_heads))
        v = K.permute_dimensions(v, [0, 3, 1, 2])  # (batch, n_heads, stations, key)
        o = tf.matmul(score, v)  # (batch, n_heads, stations, key)
        o = K.permute_dimensions(o, [0, 2, 1, 3])  # (batch, stations, n_heads, key)
        o = K.reshape(o, (-1, self.stations, n_heads * d_key))
        o = K.dot(o, self.WO)
        if mask is not None:
            mask = K.expand_dims(K.cast(mask, K.floatx()), axis=-1)
            o = K.abs(o * mask)
        return o

    def compute_output_shape(self, input_shape):
        if self.att_masking:
            return input_shape[0]
        else:
            return input_shape

    def compute_mask(self, inputs, mask=None):
        if self.att_masking:
            return mask[0]
        else:
            return mask


class PointwiseFeedForward(Layer):
    def __init__(self, hidden_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros', **kwargs):
        super(PointwiseFeedForward, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        self.kernel1 = self.add_weight('kernel1', (input_shape[-1], self.hidden_dim), initializer=self.kernel_initializer)
        self.bias1 = self.add_weight('bias1', (self.hidden_dim,), initializer=self.bias_initializer)
        self.kernel2 = self.add_weight('kernel2', (self.hidden_dim, input_shape[-1]), initializer=self.kernel_initializer)
        self.bias2 = self.add_weight('bias2', (input_shape[-1],), initializer=self.bias_initializer)
        super(PointwiseFeedForward, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, mask=None):
        x = gelu(K.dot(x, self.kernel1) + self.bias1)
        x = K.dot(x, self.kernel2) + self.bias2
        if mask is not None:
            mask = K.expand_dims(K.cast(mask, K.floatx()), axis=-1)
            x *= mask  # Zero out all masked elements
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask


class LayerNormalization(Layer):
    def __init__(self, eps=1e-5, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.beta = self.add_weight('beta', input_shape[-1:], initializer=initializers.Zeros())
        self.gamma = self.add_weight('gamma', input_shape[-1:], initializer=initializers.Ones())
        super(LayerNormalization, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, mask=None):
        # Axis according to https://github.com/tensorflow/tensor2tensor/blob/05f222d27a4885550450d9ba26987f78af5f9ecd/tensor2tensor/layers/common_layers.py#L705
        m = K.mean(x, axis=-1, keepdims=True)
        s = K.mean(K.square(x - m), axis=-1, keepdims=True)
        z = (x - m) / K.sqrt(s + self.eps)
        output = self.gamma * z + self.beta
        if mask is not None:
            mask = K.expand_dims(K.cast(mask, K.floatx()), axis=-1)
            output *= mask  # Zero out all masked elements
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask


class AddEventToken(Layer):
    def __init__(self, fixed=True, init_range=None, **kwargs):
        # If fixed: Use 1 as constant to ensure that the attention in the first layer works properly
        # Else: Use learnable event token initialized to ones
        self.fixed = fixed
        self.emb = None
        self.init_range = init_range
        super(AddEventToken, self).__init__(**kwargs)

    def build(self, input_shape):
        if not self.fixed:
            if self.init_range is None:
                initializer = initializers.Ones()
            else:
                initializer = initializers.RandomUniform(minval=-self.init_range, maxval=self.init_range)
            self.emb = self.add_weight('emb', (input_shape[2],), initializer=initializer)
        super(AddEventToken, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, mask=None):
        pad = K.ones_like(x[:, :1, :])
        if self.emb is not None:
            pad *= self.emb
        x = K.concatenate([pad, x], axis=1)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1] + 1, input_shape[2]

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            return tf.pad(mask, [[0, 0], [1, 0]], mode='CONSTANT', constant_values=True)


class AddConstantToMixture(Layer):
    def __init__(self, **kwargs):
        super(AddConstantToMixture, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AddConstantToMixture, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, mask=None):
        mix, const = x
        const = K.expand_dims(const, axis=-1)
        alpha = tf.gather(mix, 0, axis=-1)
        mu = tf.gather(mix, 1, axis=-1) + const
        sigma = tf.gather(mix, 2, axis=-1)
        output = K.stack([alpha, mu, sigma], axis=-1)
        mask = self.compute_mask(x, mask)
        if mask is not None:
            mask = K.cast(mask, dtype=K.floatx())
            while mask.ndim < output.ndim:
                mask = K.expand_dims(mask, -1)
            output *= mask
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return mask
        else:
            mask1 = mask[0]
            mask2 = mask[1]
            if mask1 is None:
                return mask2
            elif mask2 is None:
                return mask1
            else:
                return tf.logical_and(mask1, mask2)


class Masking_nd(Layer):
    def __init__(self, mask_value=0., axis=-1, nodim=False, **kwargs):
        super(Masking_nd, self).__init__(**kwargs)
        self.supports_masking = True
        self.mask_value = mask_value
        self.axis = axis
        self.nodim = nodim

    def compute_mask(self, inputs, mask=None):
        if self.nodim:
            output_mask = K.not_equal(inputs, self.mask_value)
        else:
            output_mask = K.any(K.not_equal(inputs, self.mask_value), axis=self.axis)
        return output_mask

    def call(self, inputs):
        boolean_mask = K.any(K.not_equal(inputs, self.mask_value),
                             axis=self.axis, keepdims=True)
        return inputs * K.cast(boolean_mask, K.dtype(inputs))

    def compute_output_shape(self, input_shape):
        return input_shape


class GetMask(Layer):
    def __init__(self, **kwargs):
        super(GetMask, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GetMask, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape[:2]

    def compute_mask(self, inputs, mask=None):
        return mask


class StripMask(Layer):
    def __init__(self, **kwargs):
        super(StripMask, self).__init__(**kwargs)

    def build(self, input_shape):
        super(StripMask, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, mask=None):
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return None


# From: https://github.com/openai/gpt-2/blob/ac5d52295f8a1c3856ea24fb239087cc1a3d1131/src/model.py#L25
def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))


def mixture_density_loss(y_true, y_pred, eps=1e-6, d=1, mean=True, print_shapes=True):
    if print_shapes:
        print(f'True: {y_true.shape}')
        print(f'Pred: {y_pred.shape}')
    alpha = y_pred[:, :, 0]
    density = K.ones_like(y_pred[:, :, 0])  # Create an array of ones of correct size
    for j in range(d):
        mu = y_pred[:, :, j + 1]
        sigma = y_pred[:, :, j + 1 + d]
        sigma = K.maximum(sigma, eps)
        density *= 1 / (np.sqrt(2 * np.pi) * sigma) * K.exp(-(y_true[:, j] - mu) ** 2 / (2 * sigma ** 2))
    density *= alpha
    density = K.sum(density, axis=1)
    density += eps
    loss = - K.log(density)

    if mean:
        return K.mean(loss)
    else:
        return loss


def time_distributed_loss(y_true, y_pred, loss_func, norm=1, mean=True, summation=True, kwloss={}):
    seq_length = y_pred.shape[1]
    y_true = K.reshape(y_true, (-1, (y_pred.shape[-1] - 1) // 2, 1))
    y_pred = K.reshape(y_pred, (-1, y_pred.shape[-2], y_pred.shape[-1]))
    loss = loss_func(y_true, y_pred, **kwloss)
    loss = K.reshape(loss, (-1, seq_length))

    if mean:
        return K.mean(loss)

    loss /= norm
    if summation:
        loss = K.sum(loss)

    return loss


class GlobalMaxPooling1DMasked(GlobalMaxPooling1D):
    def call(self, x, mask=None):
        pseudo_infty = 1000
        if mask is None:
            # Ensure that the mask is not the maximum value any more
            mask = K.expand_dims(K.cast(mask, K.floatx()), axis=-1)
            x = x - mask * pseudo_infty
            return K.max(x, axis=1)
        else:
            return super().call(x)

    def compute_mask(self, inputs, mask=None):
        return None


def build_transformer_model(max_stations,
                            waveform_model_dims=(500, 500, 500),
                            output_mlp_dims=(150, 100, 50, 30, 10),
                            output_location_dims=(150, 100, 50, 50, 50),
                            wavelength=((0.01, 10), (0.01, 10), (0.01, 10)),
                            mad_params={"n_heads": 10,
                                        "att_dropout": 0.0,
                                        "initializer_range": 0.02
                                        },
                            ffn_params={'hidden_dim': 1000},
                            transformer_layers=6,
                            hidden_dropout=0.0,
                            activation='relu',
                            n_pga_targets=0,
                            location_mixture=5,
                            pga_mixture=5,
                            magnitude_mixture=5,
                            borehole=False,
                            bias_mag_mu=1.8,
                            bias_mag_sigma=0.2,
                            bias_loc_mu=0,
                            bias_loc_sigma=1,
                            event_token_init_range=None,
                            dataset_bias=False,
                            n_datasets=None,
                            no_event_token=False,
                            trace_length=3000,
                            downsample=5,
                            rotation=None,
                            rotation_anchor=None,
                            skip_transformer=False,
                            alternative_coords_embedding=False,
                            **kwargs):
    if kwargs:
        print(f'Warning: Unused model parameters: {", ".join(kwargs.keys())}')

    emb_dim = waveform_model_dims[-1]
    mad_params = mad_params.copy()  # Avoid modifying the input dicts
    ffn_params = ffn_params.copy()

    if 'initializer_range' in mad_params:
        r = mad_params['initializer_range']
        mad_params['kernel_initializer'] = initializers.RandomUniform(minval=-r, maxval=r)
        del mad_params['initializer_range']

    #   Single station model
    if borehole:
        input_shape = (trace_length, 6)
        metadata_shape = (4,)
    else:
        input_shape = (trace_length, 3)
        metadata_shape = (3,)
    waveform_model = NormalizedScaleEmbedding(input_shape, downsample=downsample, activation=activation,
                                              mlp_dims=waveform_model_dims)
    mlp_mag_single_station = MLP((waveform_model.output_shape[1],), output_mlp_dims, activation=activation)
    output_model_single_station = MixtureOutput((output_mlp_dims[-1],), 5, name='magnitude',
                                                bias_mu=bias_mag_mu, bias_sigma=bias_mag_sigma)

    waveform_inp_single_station = Input(shape=input_shape)
    emb = waveform_model(waveform_inp_single_station)
    emb = mlp_mag_single_station(emb)
    out = output_model_single_station(emb)

    single_station_model = Model(waveform_inp_single_station, out)

    #   Event model

    if n_pga_targets:
        att_masking = True
        mad_params['att_masking'] = True
    else:
        att_masking = False
        mad_params['att_masking'] = False

    if not no_event_token:
        transformer_max_stations = max_stations + 1 + n_pga_targets
    else:
        transformer_max_stations = max_stations + n_pga_targets

    if not skip_transformer:
        transformer = Transformer(max_stations=transformer_max_stations, emb_dim=emb_dim, att_masking=att_masking,
                                  layers=transformer_layers, hidden_dropout=hidden_dropout, mad_params=mad_params,
                                  ffn_params=ffn_params)

    mlp_mag = MLP((emb_dim,), output_mlp_dims, activation=activation)
    output_model = MixtureOutput((output_mlp_dims[-1],), magnitude_mixture, bias_mu=bias_mag_mu,
                                 bias_sigma=bias_mag_sigma)

    mlp_loc = MLP((emb_dim,), output_location_dims, activation=activation)
    output_model_loc = MixtureOutput((output_location_dims[-1],), location_mixture, d=3, bias_mu=bias_loc_mu,
                                     bias_sigma=bias_loc_sigma, activation='linear')

    mlp_pga = MLP((emb_dim,), output_mlp_dims, activation=activation)
    output_model_pga = MixtureOutput((output_mlp_dims[-1],), pga_mixture, activation='linear', bias_mu=-5, bias_sigma=1)

    waveform_inp = Input(shape=(max_stations,) + input_shape)
    metadata_inp = Input(shape=(max_stations,) + metadata_shape)

    waveforms_masked = Masking_nd(0, (2, 3))(waveform_inp)
    coords_masked = Masking(0)(metadata_inp)

    waveforms_emb = TimeDistributed(waveform_model)(waveforms_masked)
    waveforms_emb = LayerNormalization()(waveforms_emb)

    if not alternative_coords_embedding:
        coords_emb = PositionEmbedding(wavelengths=wavelength, emb_dim=emb_dim, borehole=borehole,
                                       rotation=rotation, rotation_anchor=rotation_anchor)(coords_masked)

        emb = Add()([waveforms_emb, coords_emb])
    else:
        emb = Concatenate(axis=-1)([waveforms_emb, coords_masked])

    if not (skip_transformer or no_event_token):
        emb = AddEventToken(fixed=False, init_range=event_token_init_range)(emb)

    if n_pga_targets:
        pga_targets_inp = Input(shape=(n_pga_targets, 3))
        pga_targets_masked = Masking(0)(pga_targets_inp)
        pga_emb = PositionEmbedding(wavelengths=wavelength, emb_dim=emb_dim, borehole=borehole,
                                    rotation=rotation, rotation_anchor=rotation_anchor)(pga_targets_masked)
        att_mask = Input(tensor=K.concatenate([K.ones_like(emb[:, :, 0], dtype=bool),
                                               K.zeros_like(pga_emb[:, :, 0], dtype=bool)], axis=1))
        emb = Concatenate(axis=1)([emb, pga_emb])
        emb = transformer([emb, att_mask])
    else:
        if skip_transformer:
            mlp_input_length = emb_dim
            if alternative_coords_embedding:
                mlp_input_length += metadata_shape[0]

            emb = TimeDistributed(MLP((mlp_input_length,), [emb_dim, emb_dim], activation=activation))(emb)
            emb = GlobalMaxPooling1DMasked()(emb)
        else:
            emb = transformer(emb)

    if not no_event_token:
        if skip_transformer:
            event_emb = emb
        else:
            event_emb = Lambda(lambda x: x[:, 0, :])(emb)  # Select event embedding

        mag_embedding = mlp_mag(event_emb)
        out = output_model(mag_embedding)

        loc_embedding = mlp_loc(event_emb)
        out_loc = output_model_loc(loc_embedding)

    if n_pga_targets:
        pga_emb = Lambda(lambda x: x[:, -n_pga_targets:, :])(emb)  # Select embeddings for pga
        pga_emb = TimeDistributed(mlp_pga)(pga_emb)
        output_pga = TimeDistributed(output_model_pga, name='pga')(pga_emb)

    if dataset_bias:
        assert n_datasets is not None
        dataset = Input(shape=(1,))
        dataset_embedding = Embedding(n_datasets, 1, input_length=1)
        dataset_bias_term = dataset_embedding(dataset)
        dataset_bias_term = Flatten()(dataset_bias_term)
        dataset_bias_term = Lambda(lambda x: K.squeeze(x, -1))(dataset_bias_term)
        out = AddConstantToMixture()([out, dataset_bias_term])

    # Name output
    if not no_event_token:
        out = Lambda(lambda x: x, name='magnitude')(out)
        out_loc = Lambda(lambda x: x, name='location')(out_loc)

    inputs = [waveform_inp, metadata_inp]
    outputs = []
    if not no_event_token:
        outputs += [out, out_loc]

    if n_pga_targets:
        inputs += [pga_targets_inp, att_mask]
        outputs += [output_pga]

    if dataset_bias:
        inputs += [dataset]

    full_model = Model(inputs, outputs)

    return single_station_model, full_model


class EnsembleEvaluateModel:
    def __init__(self, config, max_ensemble_size=None, loss_limit=None):
        self.config = config
        self.ensemble = config.get('ensemble', 1)
        true_ensemble_size = self.ensemble
        if max_ensemble_size is not None:
            self.ensemble = min(self.ensemble, max_ensemble_size)
        self.models = []
        for ens_id in range(self.ensemble):
            model_params = config['model_params'].copy()
            if config['training_params'].get('ensemble_rotation', False):
                # Rotated by angles between 0 and pi/4
                model_params['rotation'] = np.pi / 4 * ens_id / (true_ensemble_size - 1)
            self.models += [build_transformer_model(**model_params)[1]]
        self.loss_limit = loss_limit

    def predict_generator(self, generator, **kwargs):
        preds = [model.predict_generator(generator, **kwargs) for model in self.models]
        return self.merge_preds(preds)

    def predict(self, inputs):
        preds = [model.predict(inputs) for model in self.models]
        return self.merge_preds(preds)

    @staticmethod
    def merge_preds(preds):
        merged_preds = []

        if isinstance(preds[0], list):
            iter = range(len(preds[0]))
        else:
            iter = [-1]

        for i in iter:  # Iterate over mag, loc, pga, ...
            if i != -1:
                pred_item = np.concatenate([x[i] for x in preds], axis=-2)
            else:
                pred_item = np.concatenate(preds, axis=-2)
            if len(pred_item.shape) == 3:
                pred_item[:, :, 0] /= np.sum(pred_item[:, :, 0], axis=-1, keepdims=True)
            elif len(pred_item.shape) == 4:
                pred_item[:, :, :, 0] /= np.sum(pred_item[:, :, :, 0], axis=-1, keepdims=True)
            else:
                raise ValueError("Encountered prediction of unexpected shape")
            merged_preds += [pred_item]

        if len(merged_preds) == 1:
            return merged_preds[0]
        else:
            return merged_preds

    def load_weights(self, weights_path):
        tmp_models = self.models
        self.models = []
        removed_models = 0
        for ens_id, model in enumerate(tmp_models):
            if self.loss_limit is not None:
                hist_path = os.path.join(weights_path, f'{ens_id}', 'hist.pkl')
                with open(hist_path, 'rb') as f:
                    hist = pickle.load(f)
                if np.min(hist['val_loss']) > self.loss_limit:
                    removed_models += 1
                    continue

            tmp_weights_path = os.path.join(weights_path, f'{ens_id}')
            weight_file = sorted([x for x in os.listdir(tmp_weights_path) if x[:5] == 'event'])[-1]
            weight_file = os.path.join(tmp_weights_path, weight_file)
            model.load_weights(weight_file)
            self.models += [model]

        if removed_models > 0:
            print(f'Removed {removed_models} models not fulfilling loss limit')
