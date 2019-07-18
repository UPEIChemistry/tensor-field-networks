"""
Module containing basic utility functions for the TFN layers
"""

import tensorflow as tf
from tensorflow.python.keras import backend as K


# FIXME: This needs to be written to be more general to RO types and static/instance calls
def inputs_to_dict(call):

    def get_input_dict(*args, **kwargs):
        if len(args) < 2:
            inputs = args[0]
            static_call = True
        else:
            instance, inputs = args[0], args[1]
            static_call = False
        if not isinstance(inputs, list):
            inputs = [inputs]
        input_dict = {0: [], 1: []}
        for tensor in inputs:
            if int(tensor.shape[-1]) == 1:
                input_dict[0].append(tensor)
            else:
                input_dict[1].append(tensor)

        # Pull out keys with empty lists
        input_dict = {k: v for k, v in input_dict.items() if len(v) != 0}
        if static_call:
            return call(input_dict, **kwargs)
        else:
            return call(instance, input_dict, **kwargs)

    return get_input_dict


def shapes_to_dict(build):

    def get_shape_dict(*args, **kwargs):
        instance, shapes = args[0], args[1]
        if not isinstance(shapes, list):
            shapes = [shapes]
        shape_dict = {0: [], 1: []}
        for shape in shapes:
            if shape[-1] == 1:
                shape_dict[0].append(shape)
            else:
                shape_dict[1].append(shape)

        # Pull out keys with empty lists
        shape_dict = {k: v for k, v in shape_dict.items() if len(v) != 0}
        build(instance, shape_dict, **kwargs)

    return get_shape_dict


def shifted_softplus(x):
    return tf.math.log(0.5 * tf.exp(x) + 0.5)


def norm_with_epsilon(x, axis=None, keepdims=False):
    return tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(x), axis=axis, keepdims=keepdims), K.epsilon()))
