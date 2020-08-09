#!/usr/bin/env python


import numpy as np
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras import backend as K
from keras import initializers
from keras.utils import conv_utils
from numpy.random import RandomState
from keras.layers import Layer, InputSpec
from keras.initializers import Initializer
from keras import activations, initializers, regularizers, constraints
from keras.layers import Lambda, Layer, InputSpec, Convolution1D, Convolution2D, add, multiply, Activation, Input, concatenate


class QuaternionConv(Layer):

    def __init__(self, rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(QuaternionConv, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = 'channels_last' if rank == 1 else K.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis] // 3
        self.kernel_shape = self.kernel_size + (input_dim, self.filters)

        kern_init = QuaternionInit(
            kernel_size=self.kernel_size,
            input_dim=input_dim,
            weight_dim=self.rank,
            nb_filters=self.filters,
        )

        self.kernel = self.add_weight(
            shape=self.kernel_shape,
            initializer=kern_init,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True
        )

        if self.use_bias:
            bias_shape = (3 * self.filters,)
            self.bias = self.add_weight(
                shape=bias_shape,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True
            )

        else:
            self.bias = None

        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim * 3})
        self.built = True

    def call(self, inputs):
        channel_axis = 1 if self.data_format == 'channels_first' else -1
        input_dim = K.shape(inputs)[channel_axis] // 3
        if self.rank == 1:
            f_phase = self.kernel[:, :, :self.filters]
            f_modulus = self.kernel[:, :, self.filters:]

        elif self.rank == 2:
            f_phase = self.kernel[:, :, :, :self.filters]
            f_modulus = self.kernel[:, :, :, self.filters:]

        elif self.rank == 3:
            f_phase = self.kernel[:, :, :, :, :self.filters]
            f_modulus = self.kernel[:, :, :, :, self.filters:]

        f_phase1 = tf.cos(f_phase)
        f_phase2 = tf.sin(f_phase) * (3 ** 0.5 / 3)
        convArgs = {"strides": self.strides[0] if self.rank == 1 else self.strides,
                    "padding": self.padding,
                    "data_format": self.data_format,
                    "dilation_rate": self.dilation_rate[0] if self.rank == 1 else self.dilation_rate}
        convFunc = {1: K.conv1d,
                    2: K.conv2d,
                    3: K.conv3d}[self.rank]

        f1 = (K.pow(f_phase1, 2) - K.pow(f_phase2, 2)) * f_modulus
        f2 = (2 * (K.pow(f_phase2, 2) - f_phase2 * f_phase1)) * f_modulus
        f3 = (2 * (K.pow(f_phase2, 2) + f_phase2 * f_phase1)) * f_modulus
        f4 = (2 * (K.pow(f_phase2, 2) + f_phase2 * f_phase1)) * f_modulus
        f5 = (K.pow(f_phase1, 2) - K.pow(f_phase2, 2)) * f_modulus
        f6 = (2 * (K.pow(f_phase2, 2) - f_phase2 * f_phase1)) * f_modulus
        f7 = (2 * (K.pow(f_phase2, 2) - f_phase2 * f_phase1)) * f_modulus
        f8 = (2 * (K.pow(f_phase2, 2) + f_phase2 * f_phase1)) * f_modulus
        f9 = (K.pow(f_phase1, 2) - K.pow(f_phase2, 2)) * f_modulus

        f1._keras_shape = self.kernel_shape
        f2._keras_shape = self.kernel_shape
        f3._keras_shape = self.kernel_shape
        f4._keras_shape = self.kernel_shape
        f5._keras_shape = self.kernel_shape
        f6._keras_shape = self.kernel_shape
        f7._keras_shape = self.kernel_shape
        f8._keras_shape = self.kernel_shape
        f9._keras_shape = self.kernel_shape
        f_phase1._keras_shape = self.kernel_shape
        f_phase2._keras_shape = self.kernel_shape

        matrix1 = K.concatenate([f1, f2, f3], axis=-2)
        matrix2 = K.concatenate([f4, f5, f6], axis=-2)
        matrix3 = K.concatenate([f7, f8, f9], axis=-2)
        matrix = K.concatenate([matrix1, matrix2, matrix3], axis=-1)
        matrix._keras_shape = self.kernel_size + (3 * input_dim, 3 * self.filters)

        output = convFunc(inputs, matrix, **convArgs)

        if self.use_bias:
            output = K.bias_add(
                output,
                self.bias,
                data_format=self.data_format
            )

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (3 * self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + (3 * self.filters,) + tuple(new_space)

    def get_config(self):
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(QuaternionConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class QuaternionConv1D(QuaternionConv):

    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 seed=None,
                 **kwargs):
        super(QuaternionConv1D, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def get_config(self):
        config = super(QuaternionConv1D, self).get_config()
        config.pop('rank')
        config.pop('data_format')
        return config


class QuaternionConv2D(QuaternionConv):

    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 seed=None,
                 **kwargs):
        super(QuaternionConv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def get_config(self):
        config = super(QuaternionConv2D, self).get_config()
        config.pop('rank')
        return config


class QuaternionConv3D(QuaternionConv):

    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1, 1),
                 activation=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 seed=None,
                 **kwargs):
        super(QuaternionConv3D, self).__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def get_config(self):
        config = super(QuaternionConv3D, self).get_config()
        config.pop('rank')
        return config

class QuaternionDense(Layer):

    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(QuaternionDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=2)
        self.supports_masking = True

    def build(self, input_shape):

        assert len(input_shape) == 2
        assert input_shape[-1] % 2 == 0
        input_dim = input_shape[-1] // 3
        data_format = K.image_data_format()
        kernel_shape = (input_dim, self.units)
        fan_in, fan_out = initializers._compute_fans(
            kernel_shape,
            data_format=data_format
        )
        s = np.sqrt(1. / fan_in)

        def init_phase(shape, dtype=None):
            return np.random.normal(
                size=kernel_shape,
                loc=0,
                scale=np.pi / 2,
            )

        def init_modulus(shape, dtype=None):
            return np.random.normal(
                size=kernel_shape,
                loc=0,
                scale=s
            )

        phase_init = init_phase
        modulus_init = init_modulus

        self.phase_kernel = self.add_weight(
            shape=kernel_shape,
            initializer=phase_init,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
        self.modulus_kernel = self.add_weight(
            shape=kernel_shape,
            initializer=modulus_init,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(3 * self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )
        else:
            self.bias = None

        self.input_spec = InputSpec(ndim=2, axes={-1: 3 * input_dim})
        self.built = True

    def call(self, inputs):
        input_shape = K.shape(inputs)
        input_dim = input_shape[-1] // 3
        phase_input = inputs[:, :input_dim]
        modulus_input = inputs[:, input_dim:]

        f_phase = self.phase_kernel
        f_phase1 = tf.cos(f_phase)
        f_phase2 = tf.sin(f_phase) * (3 ** 0.5 / 3)
        f_modulus = self.modulus_kernel

        f1 = (K.pow(f_phase1, 2) - K.pow(f_phase2, 2)) * f_modulus
        f2 = (2 * (K.pow(f_phase2, 2) - f_phase2 * f_phase1)) * f_modulus
        f3 = (2 * (K.pow(f_phase2, 2) + f_phase2 * f_phase1)) * f_modulus
        f4 = (2 * (K.pow(f_phase2, 2) + f_phase2 * f_phase1)) * f_modulus
        f5 = (K.pow(f_phase1, 2) - K.pow(f_phase2, 2)) * f_modulus
        f6 = (2 * (K.pow(f_phase2, 2) - f_phase2 * f_phase1)) * f_modulus
        f7 = (2 * (K.pow(f_phase2, 2) - f_phase2 * f_phase1)) * f_modulus
        f8 = (2 * (K.pow(f_phase2, 2) + f_phase2 * f_phase1)) * f_modulus
        f9 = (K.pow(f_phase1, 2) - K.pow(f_phase2, 2)) * f_modulus

        matrix1 = K.concatenate([f1, f2, f3], axis=-1)
        matrix2 = K.concatenate([f4, f5, f6], axis=-1)
        matrix3 = K.concatenate([f7, f8, f9], axis=-1)
        matrix = K.concatenate([matrix1, matrix2, matrix3], axis=0)

        output = K.dot(inputs, matrix)

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = 3 * self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        }
        base_config = super(QuaternionDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class QuaternionInit(Initializer):

    def __init__(self, kernel_size, input_dim,
                 weight_dim, nb_filters=None,
                 criterion='he', seed=None):

        assert len(kernel_size) == weight_dim and weight_dim in {0, 1, 2, 3}
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.input_dim = input_dim
        self.weight_dim = weight_dim
        self.criterion = criterion

    def __call__(self, shape, dtype=None):

        if self.nb_filters is not None:
            kernel_shape = tuple(self.kernel_size) + (int(self.input_dim), self.nb_filters)
        else:
            kernel_shape = (int(self.input_dim), self.kernel_size[-1])

        fan_in, fan_out = initializers._compute_fans(
            tuple(self.kernel_size) + (self.input_dim, self.nb_filters)
        )

        if self.criterion == 'glorot':
            s = 1. / (fan_in + fan_out)
        elif self.criterion == 'he':
            s = 1. / fan_in
        else:
            raise ValueError('Invalid criterion: ' + self.criterion)
        rng = RandomState(1337)

        modulus = rng.uniform(low=-np.sqrt(s) * np.sqrt(3), high=np.sqrt(s) * np.sqrt(3), size=kernel_shape)

        phase = rng.uniform(low=-np.pi / 2, high=np.pi / 2, size=kernel_shape)

        wm = modulus
        wp = phase
        weight = np.concatenate([wp, wm], axis=-1)

        return weight
