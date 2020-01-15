import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

from atomic_images.layers import (OneHot, CosineBasis, ShiftedCosineBasis, GaussianBasis,
                                  DistanceMatrix)

from .. import utils


class Preprocessing(Layer):
    """
    Convenience layer for obtaining required tensors from cartesian and point type tensors.
    Defaults to gaussians for image basis functions.

    Input:
        cartesian positions (batch, points, 3)
        point types (batch, point_type)
    Output:
        one_hot (batch, points, depth)
        image (batch, points, points, basis_functions)
        vectors (batch, points, points, 3)

    :param max_z: int. Total number of point types + 1 (for 0 type points)
    :param basis_config: dict. Contains: 'width' which specifies the size of the gaussian basis
    functions, 'spacing' which defines the size of the grid, 'min_value' which specifies the
    beginning point probed by the grid, and 'max_value' which defines the end point of the grid.
    """
    def __init__(self,
                 max_z,
                 basis_config=None,
                 basis_type='gaussian',
                 **kwargs):
        super().__init__(**kwargs)
        self.max_z = max_z
        if basis_config is None:
            basis_config = {
                'width': 0.2, 'spacing': 0.2, 'min_value': -1.0, 'max_value': 15.0
            }
        self.basis_config = basis_config
        self.basis_type = basis_type
        self.one_hot = OneHot(self.max_z)
        if self.basis_type == 'cosine':
            basis_function = CosineBasis(**self.basis_config)
        elif self.basis_type == 'shifted_cosine':
            basis_function = ShiftedCosineBasis(**self.basis_config)
        else:
            basis_function = GaussianBasis(**self.basis_config)
        self.basis_function = basis_function
        self.distance_matrix = DistanceMatrix()
        self.unit_vectors = UnitVectors()

    def call(self, inputs, **kwargs):
        r, z = inputs
        return [
            self.one_hot(z),
            self.basis_function(self.distance_matrix(r)),
            self.unit_vectors(r)
        ]

    def get_config(self):
        base = super().get_config()
        updates = dict(
            max_z=self.max_z,
            basis_config=self.basis_config
        )
        return {**base, **updates}

    def compute_output_shape(self, input_shape):
        r, _ = input_shape
        mols, atoms, _ = r
        return [
            tf.TensorShape([mols, atoms, self.max_z]),
            tf.TensorShape([mols, atoms, atoms, self.basis_function._n_centers]),
            tf.TensorShape([mols, atoms, atoms, 3])
        ]


class UnitVectors(Layer):
    """
    Input:
        cartesian positions (..., batch, points, 3)
    Output:
        unit vectors between every point in every batch (..., batch, point, point, 3)
    """

    def call(self, inputs, **kwargs):
        i = K.expand_dims(inputs, axis=-2)
        j = K.expand_dims(inputs, axis=-3)
        v = i - j
        den = utils.norm_with_epsilon(v, axis=-1, keepdims=True)
        return v / den
