import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

from tfn.layers import Convolution, Preprocessing, RadialFactory


class TestMolecularConvolution:
    def test_model_get_correct_num_trainable_weights(self, molecular_conv_inputs_and_targets, molecular_conv_model):
        inputs, targets = molecular_conv_inputs_and_targets
        model = molecular_conv_model
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(inputs, targets, epochs=2)
        assert len(model.trainable_weights) == 60  # Same as default conv model

    def test_dummy_atom_masking_masks_correct_atoms(self, random_features_and_targets, molecular_conv_model):
        r = np.random.rand(2, 10, 3).astype('float32')
        z = np.array([[0, 5, 3, 4, 5, 2, 3, 1, 0, 0],
                     [1, 1, 2, 3, 4, 4, 0, 0, 0, 0]])
        inputs, targets = random_features_and_targets
        points = [x.numpy() for x in Preprocessing(6)([r, z])]
        model = molecular_conv_model
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(points + inputs, points + targets, epochs=2)
        preds = model.predict(points + inputs)
        scalar = np.squeeze(preds[3])
        statements = [
            np.sum(scalar[0, 0]) == 0.,
            np.sum(scalar[0, 1]) != 0.,
            np.sum(scalar[0, -2]) == 0.,
            np.sum(scalar[0, -1]) == 0.,
            np.sum(scalar[1, -4]) == 0.,
            np.sum(scalar[1, -3]) == 0.,
            np.sum(scalar[1, -2]) == 0.,
            np.sum(scalar[1, -1]) == 0.
        ]
        assert all(statements)


class TestDefaultRadialFactory:
    def test_model_get_correct_num_trainable_weights(self, default_conv_inputs_and_targets, default_conv_model):
        inputs, targets = default_conv_inputs_and_targets
        model = default_conv_model
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=inputs, y=targets, epochs=2)
        # 4 tensors per filter, 4 filters per block, 4 extra tensors per block, 3 layers == 60 total
        assert len(model.trainable_weights) == 60

    def test_modified_si_correct_output_shape(self, default_conv_inputs_and_targets, default_conv_model):
        class ModifiedConvModel(default_conv_model.__class__):
            def __init__(self,
                         **kwargs):
                super().__init__(**kwargs)
                self.conv1 = Convolution(si_units=32)
                self.conv2 = Convolution(si_units=64)
                self.conv3 = Convolution(si_units=16)

        inputs, targets = default_conv_inputs_and_targets
        model = ModifiedConvModel()
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=inputs, y=targets, epochs=2)
        _ = model.predict(inputs)


class TestPassedRadialFactory:
    def test_input_filter_specific_radial_get_correct_num_trainable_weights(self,
                                                                            default_conv_inputs_and_targets,
                                                                            default_conv_model):
        class MyFactory(RadialFactory):
            def get_radial(self, feature_dim, input_order=None, filter_order=None):

                if input_order == 0:
                    if filter_order == 0:
                        return Sequential([
                            Dense(32),
                            Dense(feature_dim)
                        ])
                    elif filter_order == 1:
                        return Sequential([
                            Dense(32),
                            Dense(16),
                            Dense(feature_dim)
                        ])
                elif input_order == 1:
                    if filter_order == 0:
                        return Sequential([
                            Dense(feature_dim)
                        ])
                    elif filter_order == 1:
                        return Sequential([
                            Dense(64),
                            Dense(32),
                            Dense(16),
                            Dense(feature_dim)
                        ])

        class MyModel(default_conv_model.__class__):
            def __init__(self,
                         **kwargs):
                super().__init__(**kwargs)
                self.conv1 = Convolution(radial_factory=MyFactory())
                self.conv2 = Convolution(radial_factory=MyFactory())
                self.conv3 = Convolution(radial_factory=MyFactory())

        inputs, targets = default_conv_inputs_and_targets
        model = MyModel()
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=inputs, y=targets, epochs=2)
        # 20 filter weights and 4 extra weights per block, 3 layers == 72 total tensors
        assert len(model.trainable_weights) == 72

    # Only can have a shared radial if input features all have the SAME feature_dim
    def test_shared_radial_get_correct_num_trainable_weights(self, random_cartesians_and_z, default_conv_model):
        class MyFactory(RadialFactory):
            def __init__(self):
                self.model = Sequential([
                    Dense(16),
                    Dense(16)
                ])

            def get_radial(self, feature_dim, input_order=None, filter_order=None):
                return self.model

        class MyModel(default_conv_model.__class__):
            def __init__(self,
                         **kwargs):
                super().__init__(**kwargs)
                self.conv1 = Convolution(radial_factory=MyFactory())
                self.conv2 = Convolution(radial_factory=MyFactory())
                self.conv3 = Convolution(radial_factory=MyFactory())
        one_hot, rbf, vectors = Preprocessing(5)(random_cartesians_and_z)
        inputs = [
            rbf.numpy(),
            vectors.numpy(),
            np.random.rand(2, 10, 16, 1).astype('float32'),
            np.random.rand(2, 10, 16, 3).astype('float32')
        ]
        targets = [
            rbf.numpy(),
            vectors.numpy(),
            np.random.rand(2, 10, 16, 1).astype('float32'),
            np.random.rand(2, 10, 16, 3).astype('float32')
        ]
        model = MyModel()
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=inputs, y=targets, epochs=2)
        # 16 filter weights and 4 extra weights per block, 3 layers == 60 total tensors
        assert len(model.trainable_weights) == 60


class TestVaryingOutputOrderConvolution:

    def test_multi_layer_scalar_only_output(self, random_cartesians_and_z, default_conv_model):
        class MyModel(default_conv_model.__class__):
            def __init__(self,
                         **kwargs):
                super().__init__(**kwargs)
                self.conv1 = Convolution()
                self.conv2 = Convolution()
                self.conv3 = Convolution(output_orders=[0])

            def compute_output_shape(self, input_shape):
                return [
                    tf.TensorShape([2, 10, 10, 80]),
                    tf.TensorShape([2, 10, 10, 3]),
                    tf.TensorShape([2, 10, 16, 1])
                ]

        one_hot, rbf, vectors = Preprocessing(5)(random_cartesians_and_z)
        inputs = [
            rbf.numpy(),
            vectors.numpy(),
            np.random.rand(2, 10, 16, 1).astype('float32'),
            np.random.rand(2, 10, 16, 3).astype('float32')
        ]
        targets = [
            rbf.numpy(),
            vectors.numpy(),
            np.random.rand(2, 10, 16, 1).astype('float32')
        ]
        model = MyModel()
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=inputs, y=targets, epochs=2)
        output = model.predict(inputs)
        assert len(output) == 3
        assert output[2].shape[-1] == 1

    def test_multi_layer_vector_only_output_filter_order_0_only(self, random_cartesians_and_z, default_conv_model):
        class MyModel(default_conv_model.__class__):
            def __init__(self,
                         **kwargs):
                super().__init__(**kwargs)
                self.conv1 = Convolution()
                self.conv2 = Convolution()
                self.conv3 = Convolution(output_orders=[0], max_filter_order=0)

            def compute_output_shape(self, input_shape):
                return [
                    tf.TensorShape([2, 10, 10, 80]),
                    tf.TensorShape([2, 10, 10, 3]),
                    tf.TensorShape([2, 10, 16, 1])
                ]

        one_hot, rbf, vectors = Preprocessing(5)(random_cartesians_and_z)
        inputs = [
            rbf.numpy(),
            vectors.numpy(),
            np.random.rand(2, 10, 16, 1).astype('float32'),
            np.random.rand(2, 10, 16, 3).astype('float32')
        ]
        targets = [
            rbf.numpy(),
            vectors.numpy(),
            np.random.rand(2, 10, 16, 1).astype('float32')
        ]
        model = MyModel()
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=inputs, y=targets, epochs=2)
        output = model.predict(inputs)
        assert len(output) == 3
        assert output[2].shape[-1] == 1