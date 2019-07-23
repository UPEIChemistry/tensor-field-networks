import tensorflow as tf

import tfn.utility_layers
import tfn.utils
from functools import partial


class PreprocessingBlock(tf.keras.models.Model):

    def __init__(self,
                 max_z,
                 gaussian_config=None,
                 **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.max_z = max_z
        if gaussian_config is None:
            gaussian_config = {
                'width': 0.2, 'spacing': 0.2, 'min_value': -1.0, 'max_value': 15.0
            }
        self.one_hot = partial(tf.one_hot, depth=self.max_z)
        self.dist_matrix = tfn.utility_layers.DistanceMatrix(dynamic=True)
        self.gaussian_basis = tfn.utility_layers.GaussianBasis(**gaussian_config, dynamic=True)
        self.unit_vectors = tfn.utility_layers.UnitVectors(dynamic=True)

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
