"""
Resource and complexity metrics: FLOPs, MACs, model size, TFLite params.
Adapted from functions.py.
"""

import os
import tempfile
import zipfile
import numpy as np
import tensorflow as tf


def get_flops(model, model_inputs):
    """
    Calculate FLOPs for a tf.keras.Model in inference mode.
    Adapted from original implementation.
    """
    if not isinstance(model, (tf.keras.models.Sequential, tf.keras.models.Model)):
        raise ValueError(
            "Calculating FLOPS is only supported for "
            "`tf.keras.Model` and `tf.keras.Sequential` instances."
        )

    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph,
    )

    batch_size = 1
    inputs = [
        tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype)
        for inp in model_inputs
    ]

    real_model = tf.function(model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

    run_meta = tf.compat.v1.RunMetadata()
    opts = (
        tf.compat.v1.profiler.ProfileOptionBuilder(
            tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
        )
        .with_empty_output()
        .build()
    )

    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )

    tf.compat.v1.reset_default_graph()
    return flops.total_float_ops


def get_model_size(path):
    return os.path.getsize(path) / 1024.0  # KB


def get_gzipped_model_size(path):
    """Get gzipped model size in KB."""
    fd, zipped_file = tempfile.mkstemp(".zip")
    try:
        with zipfile.ZipFile(zipped_file, "w", compression=zipfile.ZIP_DEFLATED) as f:
            f.write(path)
        size_kb = os.path.getsize(zipped_file) / 1024.0
        return size_kb
    finally:
        os.close(fd)
        if os.path.exists(zipped_file):
            os.unlink(zipped_file)


def compute_tf_mac_operations(model):
    """
    Approximate MAC count for a TF model.
    Same logic as compute_tf_mac_operations() in functions.py.
    """
    mac_count = 0

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            filters = layer.filters
            kernel_size = np.prod(layer.kernel_size)
            mac_count += filters * kernel_size

        elif isinstance(layer, tf.keras.layers.Dense):
            units = layer.units
            # Handle case where input_shape might be None or not a tuple
            if layer.input_shape and len(layer.input_shape) > 1:
                input_shape = np.prod(layer.input_shape[1:])
            else:
                # Fallback: use output shape from previous layer or skip
                continue
            mac_count += units * input_shape

        elif isinstance(layer, tf.keras.layers.LSTM):
            units = layer.units
            # Handle case where input_shape might be None
            if layer.input_shape and len(layer.input_shape) > 0:
                input_shape = layer.input_shape[-1] if isinstance(layer.input_shape, tuple) else layer.input_shape
            else:
                # Fallback: skip or use default
                continue
            mac_count += 4 * units * (units + input_shape + 1)

    return mac_count


def compute_tflite_mac_operations(interpreter):
    """
    Approximate MAC count for a TFLite model.
    Same logic as compute_tflite_mac_operations() in functions.py.
    """
    mac_count = 0

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for tensor_details in interpreter.get_tensor_details():
        tensor_name = tensor_details["name"]
        tensor_shape = tensor_details["shape"]

        if len(tensor_shape) >= 2:
            if "conv" in tensor_name.lower():
                input_shape = input_details[0]["shape"][1:]
                output_shape = output_details[0]["shape"][1:]

                input_channels = input_shape[-1]
                output_channels = output_shape[-1]

                kernel_size = np.prod(tensor_shape[1:-1])
                mac_count += input_channels * output_channels * kernel_size

            elif "dense" in tensor_name.lower():
                input_size = np.prod(input_details[0]["shape"][1:])
                output_size = np.prod(output_details[0]["shape"][1:])
                mac_count += input_size * output_size

            elif "lstm" in tensor_name.lower() and len(tensor_shape) >= 3:
                input_size = tensor_shape[1]
                hidden_size = tensor_shape[2]

                mac_count_per_step = (
                    4 * hidden_size**2 + 4 * input_size * hidden_size + 4 * hidden_size
                )
                sequence_length = input_details[0]["shape"][1]
                mac_count += mac_count_per_step * sequence_length

    return mac_count


def get_tflite_params(interpreter):
    """Get total, trainable, and non-trainable parameter counts for TFLite model.
    
    Returns:
        tuple: (total, trainable, non_trainable)
        Note: TFLite doesn't easily distinguish trainable/non-trainable, so 
        trainable=total, non_trainable=None
    """
    total = 0
    for layer in interpreter.get_tensor_details():
        shape = layer.get("shape", [])
        if len(shape) > 0:
            total += np.prod(shape)
    return int(total), int(total), None  # TFLite doesn't distinguish trainable/non-trainable easily
