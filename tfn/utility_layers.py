import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import Layer


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
        mu = tf.cast(mu, tf.float32)
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
        """

        :param inputs: cartesian tensors of shape (points, 3)
        :param kwargs:
        :return:
        """
        i = K.expand_dims(inputs, axis=-2)
        j = K.expand_dims(inputs, axis=-3)
        v = i - j
        den = K.sqrt(
            K.maximum(K.sum(K.square(v), axis=self.axis, keepdims=self.keepdims), K.epsilon())
        )
        return v / den


class DummyAtomMasking(Layer):
    """
    Masks dummy atoms (atomic number = 0 by default) with zeros

    Inputs: atomic_numbers
                Either or both in this order:
                    atomic_numbers  (batch, atoms)
                or
                    one_hot_atomic_numbers  (batch, atoms, atomic_number)
            value  (batch, atoms, ...)
    Output: value with zeroes for dummy atoms  (batch, atoms, ...)

    Args:
        atom_axes (int or iterable of int): axes to which to apply
            the masking

    Keyword Args:
        dummy_index (int): the index to mask (default: 0)
        invert_mask (bool): if True, zeroes all but the desired index rather
            than zeroeing the desired index
    """
    def __init__(self, atom_axes=1, **kwargs):
        self.invert_mask = kwargs.pop('invert_mask', False)
        self.dummy_index = kwargs.pop('dummy_index', 0)
        super(DummyAtomMasking, self).__init__(trainable=False, **kwargs)
        if isinstance(atom_axes, int):
            atom_axes = [atom_axes]
        elif isinstance(atom_axes, tuple):
            atom_axes = list(atom_axes)
        self.atom_axes = atom_axes

    def call(self, inputs, **kwargs):
        # `value` should be of shape (batch, atoms, ...)
        one_hot_atomic_numbers, value = inputs
        atomic_numbers = K.argmax(one_hot_atomic_numbers,
                                  axis=-1)

        # Form the mask that removes dummy atoms (atomic number = dummy_index)
        if self.invert_mask:
            selection_mask = K.equal(atomic_numbers, self.dummy_index)
        else:
            selection_mask = K.not_equal(atomic_numbers, self.dummy_index)
        selection_mask = K.cast(selection_mask, value.dtype)

        for axis in self.atom_axes:
            mask = selection_mask
            for _ in range(axis - 1):
                mask = K.expand_dims(mask, axis=1)
            # Add one since K.int_shape does not return batch dim
            while len(K.int_shape(value)) != len(K.int_shape(mask)):
                mask = K.expand_dims(mask, axis=-1)

            # Zeros the energies of dummy atoms
            value *= mask
        return value
