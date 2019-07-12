import tensorflow as tf

import tfn.utils
from functools import partial


class PreprocessingBlock(tf.keras.models.Model):

    def __init__(self,
                 max_z,
                 gaussian_config,
                 **kwargs):
        super().__init__(**kwargs)
        self.one_hot = partial(tf.one_hot, depth=max_z)
        self.dist_matrix = tfn.utils.DistanceMatrix()
        self.gaussian_basis = tfn.utils.GaussianBasis(**gaussian_config)
        self.unit_vectors = tfn.utils.UnitVectors()

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