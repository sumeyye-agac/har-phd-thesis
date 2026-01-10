"""
DeepConvLSTM models with attention (LM / MM / OM + CBAM).
"""

from tensorflow.keras import models, initializers
from tensorflow.keras.layers import Input, Conv2D, Permute, Reshape, LSTM, Flatten, Dense

from models.attention_layers import ChannelAttention, SpatialAttention


CONFIGS = {
    "LM": {"filt": 4, "lstm": 8},
    "MM": {"filt": 8, "lstm": 16},
    "OM": {"filt": 64, "lstm": 128},
}


def build_deepconvlstm_att(version, input_shape, num_classes, channelatt_list, spatialatt_list, seed=42):
    """
    Build DeepConvLSTM model with attention mechanisms.
    
    Args:
        version: 'LM', 'MM', or 'OM'
        input_shape: Input shape tuple (without batch dimension)
        num_classes: Number of output classes
        channelatt_list: List of length 4, channel attention reduction ratios for each conv layer (0 means no attention)
        spatialatt_list: List of length 4, spatial attention kernel sizes for each conv layer (0 means no attention)
        seed: Random seed for initialization
    
    Returns:
        Compiled Keras model
    """
    cfg = CONFIGS[version]
    filt = cfg["filt"]
    lstm_unit = cfg["lstm"]

    initializer = initializers.Orthogonal(seed=seed)
    inp = Input(shape=input_shape, name="input_shape")

    x = Conv2D(filt, (5, 1), activation="relu", kernel_initializer=initializer)(inp)
    if channelatt_list[0] != 0:
        x = ChannelAttention(filt, channelatt_list[0], name="channel_attention_1")(x)
    if spatialatt_list[0] != 0:
        x = SpatialAttention(spatialatt_list[0], name="spatial_attention_1")(x)

    x = Conv2D(filt, (5, 1), activation="relu", kernel_initializer=initializer)(x)
    if channelatt_list[1] != 0:
        x = ChannelAttention(filt, channelatt_list[1], name="channel_attention_2")(x)
    if spatialatt_list[1] != 0:
        x = SpatialAttention(spatialatt_list[1], name="spatial_attention_2")(x)

    x = Conv2D(filt, (5, 1), activation="relu", kernel_initializer=initializer)(x)
    if channelatt_list[2] != 0:
        x = ChannelAttention(filt, channelatt_list[2], name="channel_attention_3")(x)
    if spatialatt_list[2] != 0:
        x = SpatialAttention(spatialatt_list[2], name="spatial_attention_3")(x)

    x = Conv2D(filt, (5, 1), activation="relu", kernel_initializer=initializer)(x)
    if channelatt_list[3] != 0:
        x = ChannelAttention(filt, channelatt_list[3], name="channel_attention_4")(x)
    if spatialatt_list[3] != 0:
        x = SpatialAttention(spatialatt_list[3], name="spatial_attention_4")(x)

    x = Permute((2, 1, 3))(x)
    x = Reshape((int(x.shape[1]), int(x.shape[2]) * int(x.shape[3])))(x)

    x = LSTM(lstm_unit, return_sequences=True, kernel_initializer=initializer)(x)
    x = LSTM(lstm_unit, return_sequences=True, kernel_initializer=initializer)(x)
    x = Flatten()(x)
    out = Dense(num_classes)(x)

    return models.Model(inputs=inp, outputs=out)