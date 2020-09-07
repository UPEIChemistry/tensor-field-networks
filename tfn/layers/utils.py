"""
Module containing basic utility functions for the TFN layers
"""

import math
import numpy as np
import tensorflow as tf


def norm_with_epsilon(x, axis=None, keepdims=False):
    """
    Normalizes tensor `x` along `axis`.

    :param x: Tensor being normalized
    :param axis: int. Defaults to None, which normalizes the entire tensor. Axis to normalize along.
    :param keepdims: bool. Defaults to False.
    :return: Normalized tensor.
    """
    return tf.sqrt(
        tf.maximum(tf.reduce_sum(tf.square(x), axis=axis, keepdims=keepdims), 1e-7)
    )


def rotation_matrix(axis_matrix=None, theta=math.pi / 2):
    """
    Return the 3D rotation matrix associated with counterclockwise rotation about
    the given `axis` by `theta` radians.

    :param axis_matrix: np.ndarray. Defaults to [1, 0, 0], the x-axis.
    :param theta: float. Defaults to pi/2. Rotation in radians.
    """
    axis_matrix = axis_matrix or [1, 0, 0]
    axis_matrix = np.asarray(axis_matrix)
    axis_matrix = axis_matrix / math.sqrt(np.dot(axis_matrix, axis_matrix))
    a = math.cos(theta / 2.0)
    b, c, d = -axis_matrix * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def shifted_softplus(x):
    y = tf.where(x < 14.0, tf.math.softplus(tf.where(x < 14.0, x, tf.zeros_like(x))), x)
    return y - tf.math.log(2.0)


def tfn_mae(y_pred, y_true):
    loss = tf.abs(y_pred - y_true)
    return tf.reduce_mean(loss[loss != 0])
