import tensorflow as tf
from tensorflow.keras.layers import Layer

from .atomic_images import (
    OneHot,
    DistanceMatrix,
    CosineBasis,
    ShiftedCosineBasis,
    CosineCutoff,
    GaussianBasis,
    DummyAtomMasking,
)

from . import utils


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

    def __init__(self, max_z, basis_config=None, basis_type="gaussian", **kwargs):
        self.sum_points = kwargs.pop("sum_points", False)
        super().__init__(**kwargs)
        self.max_z = max_z
        if basis_config is None:
            basis_config = {
                "width": 0.2,
                "spacing": 0.2,
                "min_value": -1.0,
                "max_value": 15.0,
            }
        self.basis_config = basis_config
        self.basis_type = basis_type
        self.one_hot = OneHot(self.max_z)
        if self.basis_type == "cosine":
            basis_function = CosineBasis(**self.basis_config)
        elif self.basis_type == "shifted_cosine":
            basis_function = ShiftedCosineBasis(**self.basis_config)
        else:
            basis_function = GaussianBasis(**self.basis_config)
        self.basis_function = basis_function
        self.cutoff = CosineCutoff(cutoff=kwargs.pop("cutoff", 15.0))
        self.distance_matrix = MaskedDistanceMatrix()
        self.unit_vectors = UnitVectors(self.sum_points)

    def call(self, inputs, **kwargs):
        z, r = inputs
        one_hot = self.one_hot(z)
        dist_matrix = self.distance_matrix([one_hot, r])
        #  (batch, points, points, basis_functions)
        rbf = self.cutoff([dist_matrix, self.basis_function(dist_matrix)])
        # (batch, points, points, 3)
        vectors = self.unit_vectors(r)
        if self.sum_points:
            rbf = tf.reduce_sum(rbf, axis=-2)
        return [one_hot, rbf, vectors]

    def get_config(self):
        base = super().get_config()
        updates = dict(max_z=self.max_z, basis_config=self.basis_config)
        return {**base, **updates}

    def compute_output_shape(self, input_shape):
        _, r = input_shape
        batch, points, _ = r
        return [
            tf.TensorShape([batch, points, self.max_z]),
            tf.TensorShape([batch, points, points, self.basis_function._n_centers]),
            tf.TensorShape([batch, points, points, 3]),
        ]


class UnitVectors(Layer):
    """
    Input:
        cartesian positions (..., batch, points, 3)
    Output:
        unit vectors between every point in every batch (..., batch, point, point, 3)
    """

    def __init__(self, sum_points: bool = False, **kwargs):
        self.sum_points = sum_points
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        if self.sum_points:
            v = inputs
        else:
            i = tf.expand_dims(inputs, axis=-2)
            j = tf.expand_dims(inputs, axis=-3)
            v = i - j
        den = utils.norm_with_epsilon(v, axis=-1, keepdims=True)
        return v / den


class MaskedDistanceMatrix(DistanceMatrix):
    def call(self, inputs, **kwargs):
        one_hot, inputs = inputs
        d = super().call(inputs, **kwargs)
        return DummyAtomMasking(atom_axes=1)(
            [one_hot, DummyAtomMasking(atom_axes=2)([one_hot, d])]
        )
