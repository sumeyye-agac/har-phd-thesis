# models/base_deepconvlstm.py
"""
DeepConvLSTM model family (OM / MM / LM) used in the thesis.

Reference:
https://github.com/AniMahajan20/DeepConvLSTM-NNFL
"""

import tensorflow as tf
from tensorflow.keras import initializers, models
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, LSTM, Permute, Reshape


def deepconvlstm_hparams(architecture: str):
    if architecture == "deepconvlstmLM":
        return 4, 8
    if architecture == "deepconvlstmMM":
        return 8, 16
    return 64, 128  # "deepconvlstm" (OM)


def build_deepconvlstm(variant: str, input_shape: tuple, num_classes: int, seed: int = 42) -> tf.keras.Model:
    """Build DeepConvLSTM model with specified variant (LM, MM, OM)."""
    architecture_map = {
        "LM": "deepconvlstmLM",
        "MM": "deepconvlstmMM",
        "OM": "deepconvlstm",
    }
    architecture = architecture_map.get(variant, "deepconvlstm")
    filt, lstm_unit = deepconvlstm_hparams(architecture)

    inputs = Input(shape=input_shape, name="input_shape")
    initializer = initializers.Orthogonal(seed=seed)

    x = Conv2D(filt, kernel_size=(5, 1), activation="relu", kernel_initializer=initializer)(inputs)
    x = Conv2D(filt, kernel_size=(5, 1), activation="relu", kernel_initializer=initializer)(x)
    x = Conv2D(filt, kernel_size=(5, 1), activation="relu", kernel_initializer=initializer)(x)
    x = Conv2D(filt, kernel_size=(5, 1), activation="relu", kernel_initializer=initializer)(x)

    x = Permute((2, 1, 3))(x)
    x = Reshape((int(x.shape[1]), int(x.shape[2]) * int(x.shape[3])))(x)

    x = LSTM(lstm_unit, dropout=0.0, return_sequences=True, kernel_initializer=initializer)(x)
    x = LSTM(lstm_unit, dropout=0.0, return_sequences=True, kernel_initializer=initializer)(x)

    x = Flatten()(x)
    outputs = Dense(num_classes)(x)

    return models.Model(inputs, outputs)


def build_deepconvlstm_om(input_shape: tuple, num_classes: int) -> tf.keras.Model:
    """Build Original DeepConvLSTM (OM) model."""
    return build_deepconvlstm("OM", input_shape, num_classes)
