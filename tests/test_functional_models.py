import os
from contextlib import contextmanager
from typing import Union

import numpy as np
import tensorflow as tf
from atomic_images.layers import Unstandardization
from tensorflow.python.keras import Input, backend as K, Model
from tensorflow.python.keras.layers import Lambda, Add

from tfn.layers import Preprocessing, SelfInteraction, MolecularConvolution, RadialFactory


class Builder(object):
    def __init__(self,
                 max_z: int,
                 mu: Union[int, list] = None,
                 sigma: Union[int, list] = None,
                 trainable_offsets: bool = False,
                 embedding_units: int = 16,
                 radial_factory: Union[RadialFactory, str] = None,
                 num_layers: int = 3,
                 si_units: int = 16,
                 residual: bool = True,
                 activation: str = None,
                 dynamic: bool = True,
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
        self.dynamic = dynamic

    def build(self):
        r = Input([10, 3], dtype='float32')
        z = Input([10, ], dtype='int32')
        point_cloud = Preprocessing(self.max_z)([r, z])  # Point cloud contains one_hot, rbf, vectors
        expanded_onehot = Lambda(lambda x: K.expand_dims(x, axis=-1))(point_cloud[0])
        learned_output = SelfInteraction(self.embedding_units)(expanded_onehot)
        for i in range(self.num_layers):
            conv = MolecularConvolution(
                name='conv_{}'.format(i),
                radial_factory=self.radial_factory,
                si_units=self.si_units,
                activation=self.activation,
                dynamic=self.dynamic
            )
            if i == 0:
                learned_output = conv(point_cloud + learned_output)
                continue
            elif self.residual:
                learned_output = [Add()([x, y]) for x, y in zip(learned_output, conv(point_cloud + learned_output))]
            else:
                learned_output = conv(point_cloud + learned_output)
        output = MolecularConvolution(
            name='energy_layer',
            radial_factory=self.radial_factory,
            si_units=1,  # For molecular energy output
            activation=self.activation,
            output_orders=[0],
            dynamic=self.dynamic
        )(point_cloud + learned_output)
        output = Lambda(lambda x: K.squeeze(x, axis=-1))(output[0])
        atomic_energies = Unstandardization(self.mu, self.sigma, trainable=self.trainable_offsets)(
            [point_cloud[0], output]
        )
        molecular_energy = Lambda(lambda x: K.sum(x, axis=-2))(atomic_energies)
        return Model(inputs=[r, z], outputs=molecular_energy)


class TestSerializability:

    @staticmethod
    @contextmanager
    def temp_file(path):
        try:
            yield path
        finally:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    def test_functional_save_model(self, random_cartesians_and_z, dynamic, eager):
        e = np.random.rand(2, 1)
        builder = Builder(max_z=6, dynamic=dynamic)
        model = builder.build()
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(random_cartesians_and_z, e, epochs=2)
        pred = model.predict(random_cartesians_and_z)

        with self.temp_file('functional_test_model.h5') as model_file:
            model.save(model_file)
            new_model = tf.keras.models.load_model(model_file)
            # tf.keras.experimental.export_saved_model(model, 'test_model.tf')
            # new_model = tf.keras.experimental.load_from_saved_model('test_model.tf')
            new_pred = new_model.predict(random_cartesians_and_z)
            assert np.alltrue(pred == new_pred)
