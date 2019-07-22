import numpy as np
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tfn.layers import RadialFactory, Convolution


class TestDefaultRadialFactory:

    def test_model_get_correct_num_trainable_weights(self, random_inputs_and_targets, default_conv_model):
        inputs, targets = random_inputs_and_targets
        model = default_conv_model
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=inputs, y=targets, epochs=2)
        # 4 tensors per filter, 4 filters per block, 4 extra tensors per block, 3 blocks == 60 total
        assert len(model.trainable_weights) == 60

    def test_modified_si_correct_output_shape(self, random_inputs_and_targets, default_conv_model):
        class ModifiedConvModel(default_conv_model.__class__):
            def __init__(self,
                         **kwargs):
                super().__init__(**kwargs)
                self.conv1 = Convolution(si_units=32)
                self.conv2 = Convolution(si_units=64)
                self.conv3 = Convolution(si_units=16)

        inputs, targets = random_inputs_and_targets
        model = ModifiedConvModel()
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=inputs, y=targets, epochs=2)
        _ = model.predict(inputs)


class TestPassedRadialFactory:

    def test_input_filter_specific_radial_get_correct_num_trainable_weights(self,
                                                                            random_inputs_and_targets,
                                                                            default_conv_model):
        class MyFactory(RadialFactory):
            def get_radial(self, feature_dim, input_ro=None, filter_ro=None):

                if input_ro == 0:
                    if filter_ro == 0:
                        return Sequential([
                            Dense(32, dynamic=True),
                            Dense(feature_dim, dynamic=True)
                        ])
                    elif filter_ro == 1:
                        return Sequential([
                            Dense(32, dynamic=True),
                            Dense(16, dynamic=True),
                            Dense(feature_dim, dynamic=True)
                        ])
                elif input_ro == 1:
                    if filter_ro == 0:
                        return Sequential([
                            Dense(feature_dim, dynamic=True)
                        ])
                    elif filter_ro == 1:
                        return Sequential([
                            Dense(64, dynamic=True),
                            Dense(32, dynamic=True),
                            Dense(16, dynamic=True),
                            Dense(feature_dim, dynamic=True)
                        ])

        class MyModel(default_conv_model.__class__):
            def __init__(self,
                         **kwargs):
                super().__init__(**kwargs)
                self.conv1 = Convolution(radial_factory=MyFactory())
                self.conv2 = Convolution(radial_factory=MyFactory())
                self.conv3 = Convolution(radial_factory=MyFactory())

        inputs, targets = random_inputs_and_targets
        model = MyModel()
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=inputs, y=targets, epochs=2)
        # 20 filter weights and 4 extra weights per block, 3 blocks == 72 total tensors
        assert len(model.trainable_weights) == 72

    # Only can have a shared radial if input features all have the SAME feature_dim
    def test_shared_radial_get_correct_num_trainable_weights(self, random_rbf_and_vectors, default_conv_model):
        class MyFactory(RadialFactory):
            def __init__(self):
                self.model = Sequential([
                    Dense(16, dynamic=True),
                    Dense(16, dynamic=True)
                ])

            def get_radial(self, feature_dim, input_ro=None, filter_ro=None):
                return self.model

        class MyModel(default_conv_model.__class__):
            def __init__(self,
                         **kwargs):
                super().__init__(**kwargs)
                self.conv1 = Convolution(radial_factory=MyFactory())
                self.conv2 = Convolution(radial_factory=MyFactory())
                self.conv3 = Convolution(radial_factory=MyFactory())
        rbf, values = random_rbf_and_vectors
        inputs = [
            rbf,
            values,
            np.random.rand(2, 10, 16, 1).astype('float32'),
            np.random.rand(2, 10, 16, 3).astype('float32')
        ]
        targets = [
            rbf,
            values,
            np.random.rand(2, 10, 16, 1).astype('float32'),
            np.random.rand(2, 10, 16, 3).astype('float32')
        ]
        model = MyModel()
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=inputs, y=targets, epochs=2)
        # 16 filter weights and 4 extra weights per block, 3 blocks == 60 total tensors
        assert len(model.trainable_weights) == 60
