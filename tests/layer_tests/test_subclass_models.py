import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tfn.layers import (
    Preprocessing,
    MolecularConvolution,
    MolecularSelfInteraction,
)
from tfn.utils import rotation_matrix


# ===== Model Subclasses ===== #


class Scalar(Model):
    def __init__(self, max_z: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.max_z = max_z
        self.preprocessing = Preprocessing(max_z)
        self.embedding = MolecularSelfInteraction(16)
        self.convolution = MolecularConvolution()
        self.output_layer = MolecularConvolution(output_orders=[0], si_units=1)

    def call(self, inputs, training=None, mask=None):
        point_cloud = self.preprocessing(inputs)
        embedding = self.embedding(
            [point_cloud[0], tf.expand_dims(point_cloud[0], axis=-1)]
        )
        output = self.convolution(point_cloud + embedding)
        output = self.output_layer(point_cloud + output)
        return tf.reduce_sum(tf.reduce_sum(output[0], axis=-2), axis=-2)

    def compute_output_shape(self, input_shape):
        batch, points, _ = input_shape[0]
        return tf.TensorShape([batch, 1])

    def get_config(self):
        return dict(max_z=self.max_z,)


class Vector(Scalar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_layer = MolecularConvolution(output_orders=[1])

    def call(self, inputs, training=None, mask=None):
        point_cloud = self.preprocessing(inputs)
        embedding = self.embedding(
            [point_cloud[0], tf.expand_dims(point_cloud[0], axis=-1)]
        )
        output = self.convolution(point_cloud + embedding)
        output = self.output_layer(point_cloud + output)
        return tf.reduce_sum(output[0], axis=-2)


# ===== Tests ===== #


class TestEquivariance:
    def test_dummy_atom_masked_predicted_vectors_rotate_correctly(
        self, random_z_and_cartesians, dynamic, eager
    ):
        z, start = random_z_and_cartesians
        end = np.random.rand(2, 10, 3)
        model = Vector(dynamic=dynamic)
        model.compile(optimizer="adam", loss="mae", run_eagerly=eager)
        model.fit([z, start], end, epochs=5)
        predicted_end = model.predict([z, start])
        rot_mat = rotation_matrix([1, 0, 0], theta=np.radians(45))
        rotated_start = np.dot(start, rot_mat)
        predicted_rotated_end = model.predict([z, rotated_start])
        assert np.all(
            np.isclose(
                np.dot(predicted_end, rot_mat),
                predicted_rotated_end,
                rtol=0,
                atol=1.0e-5,
            )
        )


class TestScalars:
    def test_default_model_predict_molecular_energies(
        self, random_z_and_cartesians, dynamic, eager
    ):
        e = np.random.rand(2, 1).astype("float32")
        model = Scalar(dynamic=dynamic)
        model.compile(optimizer="adam", loss="mae", run_eagerly=eager)
        model.fit(x=random_z_and_cartesians, y=e, epochs=2)
