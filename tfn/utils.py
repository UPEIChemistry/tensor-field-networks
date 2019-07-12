"""
Module containing basic utility functions for the TFN layers
"""

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
import numpy as np


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


class DistanceMatrix(Layer):
    """
    Distance matrix layer

    Expands Cartesian coordinates into a distance matrix.

    Input: coordinates (..., atoms, 3)
    Output: distance matrix (..., atoms, atoms)
    """
    def call(self, inputs, **kwargs):
        # `inputs` should be Cartesian coordinates of shape
        #    (..., atoms, 3)
        v1 = K.expand_dims(inputs, axis=-2)
        v2 = K.expand_dims(inputs, axis=-3)

        sum_squares = K.sum(K.square(v2 - v1), axis=-1)
        sqrt = K.sqrt(sum_squares + K.epsilon())
        K.switch(sqrt >= K.epsilon(), sqrt, K.zeros_like(sqrt))
        return sqrt

    def compute_output_shape(self, positions_shape):
        return positions_shape[0:-2] + (positions_shape[-2], positions_shape[-2])


class KernelBasis(Layer):
    """Expand tensor using kernel of width=width, spacing=spacing,
    starting at min_value ending at max_value (inclusive if endpoint=True).

    Input: tensor (batch, atoms, [atoms, [atoms...])
    Output: tensor expanded into kernel basis set (batch, atoms, [atoms, [atoms...]], n_gaussians)

    Args:
        min_value (float, optional): minimum value
        max_value (float, optional): maximum value (non-inclusive)
        width (float, optional): width of kernel functions
        spacing (float, optional): spacing between kernel functions
        self_thresh (float, optional): value below which a distance is
            considered to be a self interaction (i.e. zero)
        include_self_interactions (bool, optional): whether or not to include
            self-interactions (i.e. distance is zero)
    """
    def __init__(self,
                 min_value=-1,
                 max_value=9,
                 width=0.2,
                 spacing=0.2,
                 self_thresh=1e-5,
                 include_self_interactions=True,
                 endpoint=False,
                 **kwargs):
        super(KernelBasis, self).__init__(**kwargs)
        self._n_centers = int(np.ceil((max_value - min_value) / spacing))
        self.min_value = min_value
        self.max_value = max_value
        self.spacing = spacing
        self.width = width
        self.self_thresh = self_thresh
        self.include_self_interactions = include_self_interactions
        self.endpoint = endpoint

    def call(self, inputs, **kwargs):
        in_tensor = K.expand_dims(inputs, -1)
        mu = tf.linspace(self.min_value, self.max_value, self._n_centers)

        mu_prefix_shape = tuple([1 for _ in range(len(tf.shape(in_tensor)) - 1)])
        mu = K.reshape(mu, mu_prefix_shape + (-1,))
        mu = tf.cast(mu, tf.float64)
        values = self.kernel_func(in_tensor, mu)

        if not self.include_self_interactions:
            mask = K.cast(in_tensor >= self.self_thresh, K.floatx())
            values *= mask

        return values

    def kernel_func(self, inputs, centres):
        raise NotImplementedError

    def compute_output_shape(self, in_tensor_shape):
        return in_tensor_shape + (self._n_centers,)


class GaussianBasis(KernelBasis):
    """Expand distance matrix into Gaussians of width=width, spacing=spacing,
    starting at min_value ending at max_value (inclusive if endpoint=True).

        -(x - u)^2
    exp(----------)
        2 * w^2

    where: u is linspace(min_value, max_value, ceil((max_value - min_value) / width))
           w is width

    Input: distance_matrix (batch, atoms, atoms)
    Output: distance_matrix expanded into Gaussian basis set (batch, atoms, atoms, n_centres)
    """
    def kernel_func(self, inputs, centres):
        gamma = -0.5 / (self.width ** 2)
        return tf.exp(gamma * tf.square(inputs - centres))


class UnitVectors(Layer):

    def __init__(self,
                 axis=-1,
                 keepdims=True,
                 **kwargs):
        super(UnitVectors, self).__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, inputs, **kwargs):
        i = K.expand_dims(inputs, axis=-2)
        j = K.expand_dims(inputs, axis=-3)
        v = i - j
        den = K.sqrt(
            K.maximum(K.sum(K.square(v), axis=self.axis, keepdims=self.keepdims), K.epsilon())
        )
        return v / den
