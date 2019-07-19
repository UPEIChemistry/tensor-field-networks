"""
Module containing basic utility functions for the TFN layers
"""

import tensorflow as tf
from tensorflow.python.keras import backend as K


def norm_with_epsilon(x, axis=None, keepdims=False):
    return tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(x), axis=axis, keepdims=keepdims), K.epsilon()))
