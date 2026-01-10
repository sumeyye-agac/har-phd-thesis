"""
Channel and Spatial Attention layers.
Adapted from attention.py.
"""

from tensorflow.keras.layers import (
    Layer,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Add,
    Activation,
    Multiply,
    Lambda,
    Concatenate,
    Dense,
    Conv2D,
    Reshape,
)
from tensorflow.keras import backend as K


class ChannelAttention(Layer):
    def __init__(self, filters, ratio, name=None):
        super(ChannelAttention, self).__init__(name=name)
        self.filters = filters
        self.ratio = ratio
        self.attention = None

    def build(self, input_shape):
        self.shared_layer_one = Dense(
            self.filters // self.ratio,
            activation="relu",
            kernel_initializer="he_normal",
            use_bias=True,
            bias_initializer="zeros",
        )
        self.shared_layer_two = Dense(
            self.filters,
            kernel_initializer="he_normal",
            use_bias=True,
            bias_initializer="zeros",
        )

    def call(self, inputs):
        avg_pool = GlobalAveragePooling2D()(inputs)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)

        max_pool = GlobalMaxPooling2D()(inputs)
        max_pool = Reshape((1, 1, self.filters))(max_pool)
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)

        attention1 = Add()([avg_pool, max_pool])
        self.attention = Activation("sigmoid")(attention1)

        return Multiply()([inputs, self.attention])

    def get_attention(self):
        return self.attention


class SpatialAttention(Layer):
    def __init__(self, kernel_size, name=None):
        super(SpatialAttention, self).__init__(name=name)
        self.kernel_size = kernel_size
        self.attention = None

    def build(self, input_shape):
        self.conv2d = Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            strides=1,
            padding="same",
            activation="sigmoid",
            kernel_initializer="he_normal",
            use_bias=False,
        )

    def call(self, inputs):
        avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(inputs)
        max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(inputs)
        attention1 = Concatenate(axis=3)([avg_pool, max_pool])
        self.attention = self.conv2d(attention1)
        return Multiply()([inputs, self.attention])

    def get_attention(self):
        return self.attention
