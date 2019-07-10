import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.activations import relu
import numpy as np


class HarmonicFilter(Layer):
    """
    Abstract class for generating filters.

    :param radial: Callable. The learnable bits of an equivariant filter. Radial can be any tf callable
        (model, layer, op...) that takes the RBF image of shape (atoms, atoms, rbf) as input and combines it
        in some way with weights to return a learned tensor of shape (atoms, atoms, output_dim) that, when combined
        with a tensor derived from a spherical harmonic function aligned with provided unit vectors, returns a filter.
    :param num_filters: int. How many filters the generator should output. Supports up to 2 filters, currently.
    """
    def __init__(self,
                 radial=None,
                 filter_ro=0,
                 **kwargs):
        super().__init__(**kwargs)
        if radial is None:
            radial = Sequential([
                Dense(16, activation=relu),
                Dense(16, activation=relu)
            ])
        self.radial = radial
        self.filter_ro = filter_ro

    def call(self, inputs, **kwargs):
        """Generate the filter based on provided image (and vectors, depending on requested filter rotation order).

        :param inputs: List of input tensors including image, of shape (atoms, atoms, rbf), and unit_vectors, of shape
            (atoms, atoms, 3).
        :return: tensor. HarmonicFilter of specified rotation order.
        """
        image, vectors = inputs
        if self.filter_ro == 0:
            # [N, N, output_dim, 1]
            return K.expand_dims(self.radial(image), axis=-1)
        elif self.filter_ro == 1:
            masked_radial = self.mask_radial(self.radial(image), vectors)
            # [N, N, output_dim, 3]
            return K.expand_dims(vectors, axis=-2) * K.expand_dims(masked_radial, axis=-1)
        elif self.filter_ro == 2:
            masked_radial = self.mask_radial(self.radial(image), vectors)
            # [N, N, output_dim, 5]
            return K.expand_dims(self.l2_spherical_harmonic(vectors), axis=-2) * K.expand_dims(masked_radial, axis=-1)

    @staticmethod
    def mask_radial(radial, vectors):
        norm = tf.norm(vectors, axis=-1)
        condition = K.expand_dims(norm < K.epsilon(), axis=-1)
        tile = K.tile(condition, [1, 1, radial.shape[-1]])

        # [N, N, output_dim]
        return tf.where(tile, K.zeros_like(radial), radial)

    @staticmethod
    def l2_spherical_harmonic(tensor):
        """

        :param tensor: must be of shape [atoms, atoms, 3]
        :return: tensor. Result of L2 spherical harmonic function applied to input tensor
        """
        x = tensor[:, :, 0]
        y = tensor[:, :, 1]
        z = tensor[:, :, 2]
        r2 = tf.maximum(tf.reduce_sum(tf.square(tensor), axis=-1), K.epsilon())
        # return : [N, N, 5]
        output = tf.stack([x * y / r2,
                           y * z / r2,
                           (-tf.square(x) - tf.square(y) + 2. * tf.square(z)) / (2 * np.sqrt(3) * r2),
                           z * x / r2,
                           (tf.square(x) - tf.square(y)) / (2. * r2)],
                          axis=-1)
        return output
