import tensorflow as tf
from .. import layers_v2 as layers
from functools import partial


class PreprocessingBlock(tf.keras.models.Model):

    def __init__(self,
                 max_z,
                 gaussian_config,
                 **kwargs):
        super().__init__(**kwargs)
        self.one_hot = partial(tf.one_hot, depth=max_z)
        self.dist_matrix = layers.DistanceMatrix()
        self.gaussian_basis = layers.GaussianBasis(**gaussian_config)
        self.unit_vectors = layers.UnitVectors()

    def call(self, inputs, **kwargs):
        """
        Convert cartesians and atomic_nums into required tensors
        :param inputs: list. cartesian coordinates and atomic nums, in that order
        :return: list. one_hot, rbf, and unit_vectors tensors in that order.
        """
        r, z = inputs
        return [
            self.one_hot(z),
            self.gaussian_basis(self.dist_matrix(r)),
            self.unit_vectors(r)
        ]


# TODO: Fill in layer params when layers are written!
class EquivariantBlock(tf.keras.models.Model):

    def __init__(self,
                 image,
                 unit_vectors,
                 num_filters,
                 **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Convolution(image, unit_vectors, num_filters=num_filters)
        self.concat = layers.Concatenation()
        self.si = layers.SelfInteraction(num_filters)
        self.nonlin = layers.Nonlinearity()

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.concat(x)
        x = self.si(x)
        return self.nonlin(x)
