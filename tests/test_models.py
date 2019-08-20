import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import BatchNormalizationV2

from tfn.layers import Preprocessing
from tfn.utils import rotation_matrix


class TestEquivariance:
    def test_conv_no_dummy_atoms_predicted_vectors_rotate_correctly(self,
                                                                    random_cartesians_and_z,
                                                                    vector_model_no_dummy):
        start, z = random_cartesians_and_z
        end = np.random.rand(2, 10, 3)
        model = vector_model_no_dummy
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit([start, z], end, epochs=5)
        predicted_end = model.predict([start, z])
        R = rotation_matrix([1, 0, 0], theta=np.radians(45))
        rotated_start = np.dot(start, R)
        predicted_rotated_end = model.predict([rotated_start, z])

        assert np.all(np.isclose(np.dot(predicted_end, R), predicted_rotated_end, rtol=0, atol=1.e-5))

    def test_dummy_atom_masked_predicted_vectors_rotate_correctly(self, random_cartesians_and_z, vector_model):
        start, z = random_cartesians_and_z
        end = np.random.rand(2, 10, 3)
        model = vector_model
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit([start, z], end, epochs=5)
        predicted_end = model.predict([start, z])
        R = rotation_matrix([1, 0, 0], theta=np.radians(45))
        rotated_start = np.dot(start, R)
        predicted_rotated_end = model.predict([rotated_start, z])

        assert np.all(np.isclose(np.dot(predicted_end, R), predicted_rotated_end, rtol=0, atol=1.e-5))

    def test_batch_norm_dummy_atoms_perserves_equivariance(self, random_cartesians_and_z, vector_model):
        class NormModel(vector_model.__class__):
            def call(self, inputs, training=None, mask=None):
                r, z = inputs
                one_hot, rbf, vectors = Preprocessing(self.max_z, self.gaussian_config)([r, z])
                embedding = self.embedding(tf.transpose(one_hot, perm=[0, 1, 3, 2]))
                output = self.conv1([one_hot, rbf, vectors] + embedding)
                output = [BatchNormalizationV2(axis=-2, dynamic=True)(o) for o in output]
                output = self.conv2([one_hot, rbf, vectors] + output)
                output = [BatchNormalizationV2(axis=-2, dynamic=True)(o) for o in output]
                output = self.conv3([one_hot, rbf, vectors] + output)
                output = [BatchNormalizationV2(axis=-2, dynamic=True)(o) for o in output]
                return tf.reduce_sum(
                    output[1], axis=-2
                )

        start, z = random_cartesians_and_z
        end = np.random.rand(2, 10, 3)
        model = NormModel()
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit([start, z], end, epochs=5)
        predicted_end = model.predict([start, z])
        R = rotation_matrix([1, 0, 0], theta=np.radians(45))
        rotated_start = np.dot(start, R)
        predicted_rotated_end = model.predict([rotated_start, z])

        assert np.all(np.isclose(np.dot(predicted_end, R), predicted_rotated_end, rtol=0, atol=1.e-5))


class TestEnergyModels:
    def test_default_conv_model_predict_molecular_energies(self, scalar_model):
        cartesians = np.random.rand(2, 10, 3).astype('float32')
        atomic_nums = np.random.randint(0, 5, size=(2, 10, 1))
        e = np.random.rand(2, 1).astype('float32')
        model = scalar_model
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=[cartesians, atomic_nums], y=e, epochs=2)

    def test_residual_conv_model_predicts_molecular_energy(self, scalar_model):
        class MyModel(scalar_model.__class__):
            def call(self, inputs, training=None, mask=None):
                r, z = inputs  # (batch, points, 3) and (batch, points)
                # Slice r, z for single mol
                one_hot, rbf, vectors = Preprocessing(self.max_z, self.gaussian_config)([r, z])
                embedding = self.embedding(tf.transpose(one_hot, [0, 1, 3, 2]))
                output = self.conv1([one_hot, rbf, vectors] + embedding)
                output = [x + y for x, y in zip(output, self.conv2([one_hot, rbf, vectors] + output))]
                output = [x + y for x, y in zip(output, self.conv3([one_hot, rbf, vectors] + output))]
                assert len(output) == 2  # Combining things properly
                return tf.reduce_sum(tf.reduce_sum(output[0], axis=-2), axis=-2)

        cartesians = np.random.rand(2, 10, 3).astype('float32')
        atomic_nums = np.random.randint(0, 5, size=(2, 10, 1))
        e = np.random.rand(2, 1).astype('float32')
        model = MyModel()
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=[cartesians, atomic_nums], y=e, epochs=2)
