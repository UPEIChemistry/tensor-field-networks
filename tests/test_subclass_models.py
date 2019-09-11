import json
from typing import Union
import shutil
from contextlib import contextmanager

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import BatchNormalizationV2
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tfn.layers import Preprocessing, MolecularConvolution, SelfInteraction, RadialFactory, DenseRadialFactory
from tfn.utils import rotation_matrix
from atomic_images.layers import Unstandardization


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
        embedding = self.embedding(K.expand_dims(one_hot, axis=-1))
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
        embedding = self.embedding(K.expand_dims(one_hot, axis=-1))
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
                embedding = self.embedding(
                        K.expand_dims(point_cloud[0], axis=-1),
                        pattern=[0, 1, 3, 2]
                    )
                output = self.batch1(self.conv1(point_cloud + embedding)[0])
                output = self.batch2(self.conv2(point_cloud + [output])[0])
                output = self.batch3(self.conv3(point_cloud + [output])[0])
                return K.sum(
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
        atomic_nums = np.random.randint(0, 5, size=(2, 10))
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
                embedding = self.embedding(K.permute_dimensions(one_hot, pattern=[0, 1, 3, 2]))
                output = self.conv1([one_hot, rbf, vectors] + embedding)
                output = [x + y for x, y in zip(output, self.conv2([one_hot, rbf, vectors] + output))]
                output = self.conv3([one_hot, rbf, vectors] + output)
                assert len(output) == 1  # Combining things properly
                return K.sum(K.sum(output[0], axis=-2), axis=-2)

        cartesians = np.random.rand(2, 10, 3).astype('float32')
        atomic_nums = np.random.randint(0, 5, size=(2, 10, 1))
        e = np.random.rand(2, 1).astype('float32')
        model = MyModel(dynamic=dynamic)
        model.compile(optimizer='adam', loss='mae', run_eagerly=eager)
        model.fit(x=[cartesians, atomic_nums], y=e, epochs=2)


class TestSerialization:

    @staticmethod
    @contextmanager
    def temp_file(path):
        try:
            yield path
        finally:
            shutil.rmtree(path, ignore_errors=True)

    class SerializeModel(Model):
        def __init__(self,
                     max_z: int,
                     mu: Union[int, list] = None,
                     sigma: Union[int, list] = None,
                     trainable_offsets: bool = False,
                     embedding_units: int = 16,
                     radial_factory: Union[RadialFactory, str] = DenseRadialFactory(),
                     num_layers: int = 3,
                     si_units: int = 16,
                     residual: bool = True,
                     activation: str = None,
                     **kwargs):
            super().__init__(**kwargs)
            self.max_z = max_z
            if mu is None:
                mu = np.array(
                    [
                        0.,  # Dummy atoms
                        -13.61312172,  # Hydrogens
                        -1029.86312267,  # Carbons
                        -1485.30251237,  # Nitrogens
                        -2042.61123593,  # Oxygens
                        -2713.48485589  # Fluorines
                    ]
                ).reshape((6, 1))
            if sigma is None:
                sigma = np.ones_like(mu)
            self.mu = mu
            self.sigma = sigma
            self.trainable_offsets = trainable_offsets
            self.embedding_units = embedding_units
            self.radial_factory = radial_factory
            self.num_layers = num_layers
            self.si_units = si_units
            self.residual = residual
            self.activation = activation

            self.preprocessing = Preprocessing(self.max_z)
            self.embedding = SelfInteraction(self.embedding_units)
            self.conv_layers = [
                MolecularConvolution(
                    name='conv_{}'.format(i),
                    radial_factory=self.radial_factory,
                    si_units=self.si_units,
                    activation=self.activation
                ) for i in range(self.num_layers)
            ]
            self.energy_layer = MolecularConvolution(self.radial_factory, 1,
                                                     self.activation, output_orders=[0], name='conv_energy')
            self.unstandardize = Unstandardization(self.mu, self.sigma, trainable=self.trainable_offsets)

        def call(self, inputs, training=None, mask=None):
            r, z = inputs
            point_cloud = self.preprocessing([r, z])  # Point cloud contains one_hot, rbf, vectors
            learned_output = self.embedding(K.expand_dims(point_cloud[0], axis=-1))
            for i, conv in enumerate(self.conv_layers):
                if i == 0:
                    learned_output = conv(point_cloud + learned_output)
                    continue
                elif self.residual:
                    learned_output = [x + y for x, y in zip(learned_output, conv(point_cloud + learned_output))]
                else:
                    learned_output = conv(point_cloud + learned_output)
                output = self.energy_layer(point_cloud + learned_output)
                output = K.squeeze(output[0], axis=-1)
                atomic_energies = self.unstandardize([point_cloud[0], output])
                self._update_config()
                return K.sum(atomic_energies, axis=1)

        def _update_config(self):
            configs = [json.loads(c.radial_factory.to_json()) for c in self.conv_layers]
            self.radial_config = {c: v for config in configs for c, v in config.items()}

        def compute_output_shape(self, input_shape):
            mols, atoms, _ = input_shape[0]
            return tf.TensorShape([mols, 1])

        def get_config(self):
            base = super().get_config()
            mu = self.mu
            if isinstance(mu, (np.ndarray, np.generic)):
                if len(mu.shape) > 0:
                    mu = mu.tolist()
                else:
                    mu = float(mu)

            sigma = self.sigma
            if isinstance(sigma, (np.ndarray, np.generic)):
                if len(sigma.shape) > 0:
                    sigma = sigma.tolist()
                else:
                    sigma = float(sigma)
            updates = dict(
                max_z=self.max_z,
                mu=mu,
                sigma=sigma,
                trainable_offsets=self.trainable_offsets,
                embedding_units=self.embedding_units,
                radial_factory=self.radial_factory.to_json(),
                num_layers=self.num_layers,
                si_units=self.si_units,
                residual=self.residual,
                activation=self.activation,
            )
            return {**base, **updates}

    def test_energy_model_serializes_and_loads(self, random_cartesians_and_z, dynamic, eager):
        e = np.random.rand(2, 1).astype('float32')
        model = self.SerializeModel(6, dynamic=False)
        model.compile(optimizer='adam', loss='mae')
        model.fit(random_cartesians_and_z, e, epochs=3)
        pred = model.predict(random_cartesians_and_z)
        with self.temp_file('./subclass_test_model.tf') as model_path:
            model.save(model_path)
            new_model = tf.keras.models.load_model(model_path, custom_objects={
                'SerializeModel': self.SerializeModel
            })
            new_pred = new_model.predict(random_cartesians_and_z)
            assert np.allclose(pred, new_pred, atol=100)
