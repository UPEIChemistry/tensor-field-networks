import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import BatchNormalizationV2
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tfn.layers import Preprocessing, MolecularConvolution, SelfInteraction
from tfn.utils import rotation_matrix


# ===== Model Subclasses ===== #

class ScalarModel(Model):
    def __init__(self,
                 max_z=6,
                 gaussian_config=None,
                 output_orders=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_z = max_z
        self.gaussian_config = gaussian_config
        if output_orders is None:
            output_orders = [0]
        self.output_orders = output_orders
        self.embedding = SelfInteraction(32)
        self.conv1 = MolecularConvolution()
        self.conv2 = MolecularConvolution()
        self.conv3 = MolecularConvolution(output_orders=output_orders)

    def call(self, inputs, training=None, mask=None):
        r, z = inputs  # (mols, atoms, 3) and (mols, atoms)
        # Slice r, z for single mol
        one_hot, rbf, vectors = Preprocessing(self.max_z, self.gaussian_config)([r, z])
        embedding = self.embedding(K.permute_dimensions(one_hot, [0, 1, 3, 2]))
        output = self.conv1([one_hot, rbf, vectors] + embedding)
        output = self.conv2([one_hot, rbf, vectors] + output)
        output = self.conv3([one_hot, rbf, vectors] + output)
        return K.sum(
            K.sum(
                output[0], axis=-2
            ), axis=-2
        )

    def compute_output_shape(self, input_shape):
        mols, atoms, _ = input_shape[0]
        return tf.TensorShape([mols, 1])

    def get_config(self):
        return dict(
            max_z=self.max_z,
            gaussian_config=self.gaussian_config,
            output_orders=self.output_orders
        )


class VectorModel(ScalarModel):
    def __init__(self,
                 **kwargs):
        super().__init__(output_orders=[1], **kwargs)

    def call(self, inputs, training=None, mask=None):
        r, z = inputs  # (mols, atoms, 3) and (mols, atoms)
        # Slice r, z for single mol
        one_hot, rbf, vectors = Preprocessing(self.max_z, self.gaussian_config)([r, z])
        embedding = self.embedding(K.permute_dimensions(one_hot, [0, 1, 3, 2]))
        output = self.conv1([one_hot, rbf, vectors] + embedding)
        output = self.conv2([one_hot, rbf, vectors] + output)
        output = self.conv3([one_hot, rbf, vectors] + output)
        return K.sum(
                output[0],
                axis=-2
        )

    def compute_output_shape(self, input_shape):
        mols, atoms, _ = input_shape[0]
        return tf.TensorShape([mols, atoms, 3])


# ===== Tests ===== #

class TestEquivariance:
    def test_dummy_atom_masked_predicted_vectors_rotate_correctly(self, random_cartesians_and_z, dynamic, eager):
        start, z = random_cartesians_and_z
        end = np.random.rand(2, 10, 3)
        model = VectorModel(dynamic=dynamic)
        model.compile(optimizer='adam', loss='mae', run_eagerly=eager)
        model.fit([start, z], end, epochs=5)
        predicted_end = model.predict([start, z])
        R = rotation_matrix([1, 0, 0], theta=np.radians(45))
        rotated_start = np.dot(start, R)
        predicted_rotated_end = model.predict([rotated_start, z])
        assert np.all(np.isclose(np.dot(predicted_end, R), predicted_rotated_end, rtol=0, atol=1.e-5))

    def test_batch_norm_dummy_atoms_perserves_equivariance(self, random_cartesians_and_z, dynamic, eager):
        class NormModel(VectorModel):
            def __init__(self,
                         **kwargs):
                super().__init__(**kwargs)
                self.conv1 = MolecularConvolution(output_orders=[1])
                self.conv2 = MolecularConvolution(output_orders=[1])
                self.conv3 = MolecularConvolution(output_orders=[1])
                self.batch1 = BatchNormalizationV2(axis=-2)
                self.batch2 = BatchNormalizationV2(axis=-2)
                self.batch3 = BatchNormalizationV2(axis=-2)

            def call(self, inputs, training=None, mask=None):
                r, z = inputs
                point_cloud = Preprocessing(self.max_z, self.gaussian_config)([r, z])
                embedding = self.embedding(tf.transpose(point_cloud[0], perm=[0, 1, 3, 2]))
                output = self.batch1(self.conv1(point_cloud + embedding)[0])
                output = self.batch2(self.conv2(point_cloud + [output])[0])
                output = self.batch3(self.conv3(point_cloud + [output])[0])
                return tf.reduce_sum(
                    output, axis=-2
                )

        start, z = random_cartesians_and_z
        end = np.random.rand(2, 10, 3)
        model = NormModel(dynamic=dynamic)
        model.compile(optimizer='adam', loss='mae', run_eagerly=eager)
        model.fit([start, z], end, epochs=5)
        predicted_end = model.predict([start, z])
        R = rotation_matrix([1, 0, 0], theta=np.radians(45))
        rotated_start = np.dot(start, R)
        predicted_rotated_end = model.predict([rotated_start, z])
        assert np.all(np.isclose(np.dot(predicted_end, R), predicted_rotated_end, rtol=0, atol=5e-1))


class TestEnergyModels:
    def test_default_model_predict_molecular_energies(self, dynamic, eager):
        cartesians = np.random.rand(2, 10, 3).astype('float32')
        atomic_nums = np.random.randint(0, 5, size=(2, 10, 1))
        e = np.random.rand(2, 1).astype('float32')
        model = ScalarModel(dynamic=dynamic)
        model.compile(optimizer='adam', loss='mae', run_eagerly=eager)
        model.fit(x=[cartesians, atomic_nums], y=e, epochs=2)

    def test_residual_conv_model_predicts_molecular_energy(self, dynamic, eager):
        class MyModel(ScalarModel):
            def call(self, inputs, training=None, mask=None):
                r, z = inputs  # (batch, points, 3) and (batch, points)
                # Slice r, z for single mol
                one_hot, rbf, vectors = Preprocessing(self.max_z, self.gaussian_config)([r, z])
                embedding = self.embedding(tf.transpose(one_hot, [0, 1, 3, 2]))
                output = self.conv1([one_hot, rbf, vectors] + embedding)
                output = [x + y for x, y in zip(output, self.conv2([one_hot, rbf, vectors] + output))]
                output = self.conv3([one_hot, rbf, vectors] + output)
                assert len(output) == 1  # Combining things properly
                return tf.reduce_sum(tf.reduce_sum(output[0], axis=-2), axis=-2)

        cartesians = np.random.rand(2, 10, 3).astype('float32')
        atomic_nums = np.random.randint(0, 5, size=(2, 10, 1))
        e = np.random.rand(2, 1).astype('float32')
        model = MyModel(dynamic=dynamic)
        model.compile(optimizer='adam', loss='mae', run_eagerly=eager)
        model.fit(x=[cartesians, atomic_nums], y=e, epochs=2)
