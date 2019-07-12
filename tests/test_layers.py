import tensorflow as tf
from tfn import layers, utils
import numpy as np
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Conv1D, Dense


class MyModel(Model):

    def __init__(self,
                 layer,
                 **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.layer = layer

    def call(self, inputs, training=None, mask=None):
        return self.layer(inputs)


class TestConvolutionModels:

    def test_single_conv_layer_model_default_radial_default_si_gets_trainable_weights(self):
        class FakeModel(MyModel):
            def compute_output_shape(self, input_shape):
                return [
                    tf.TensorShape([10, 16, 1]),
                    tf.TensorShape([10, 16, 3])
                ]

        rbf = np.random.rand(10, 10, 80).astype('float32')
        vectors = np.random.rand(10, 10, 3).astype('float32')
        embedding = np.random.rand(10, 16, 1).astype('float32')
        rand_y = [
            np.random.rand(10, 16, 1).astype('float32'),
            np.random.rand(10, 16, 3).astype('float32')
        ]
        model = FakeModel(layers.Convolution(rbf, vectors))
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=embedding, y=rand_y, epochs=1)

        assert model.trainable_weights

    def test_single_conv_layer_model_default_radial_default_si_fits_multiple_ro_inputs(self):
        class FakeModel(MyModel):
            def compute_output_shape(self, input_shape):
                return [
                    tf.TensorShape([10, 16, 1]),
                    tf.TensorShape([10, 16, 3])
                ]

        rbf = np.random.rand(10, 10, 80).astype('float32')
        vectors = np.random.rand(10, 10, 3).astype('float32')
        embedding = [
            np.random.rand(10, 16, 1).astype('float32'),
            np.random.rand(10, 16, 3).astype('float32')
        ]
        rand_y = [
            np.random.rand(10, 16, 1).astype('float32'),
            np.random.rand(10, 16, 3).astype('float32'),
        ]

        model = FakeModel(layers.Convolution(rbf, vectors))
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=embedding, y=rand_y, epochs=1)
        preds = model.predict(embedding)

        assert preds[0].shape == (10, 16, 1) and preds[1].shape == (10, 16, 3)

    def test_multiple_conv_layer_model_default_radial_default_si_fits_multiple_ro_inputs(self):
        class FakeModel(Model):
            def __init__(self,
                         image,
                         unit_vectors,
                         **kwargs):
                super().__init__(**kwargs)
                self.conv1 = layers.Convolution(image, unit_vectors)
                self.conv2 = layers.Convolution(image, unit_vectors)
                self.conv3 = layers.Convolution(image, unit_vectors)

            def call(self, inputs, training=None, mask=None):
                return self.conv3(self.conv2(self.conv1(inputs)))

            def compute_output_shape(self, input_shape):
                return [
                    tf.TensorShape([10, 16, 1]),
                    tf.TensorShape([10, 16, 3])
                ]

        rbf = np.random.rand(10, 10, 80).astype('float32')
        vectors = np.random.rand(10, 10, 3).astype('float32')
        embedding = [
            np.random.rand(10, 16, 1).astype('float32'),
            np.random.rand(10, 16, 3).astype('float32')
        ]
        rand_y = [
            np.random.rand(10, 16, 1).astype('float32'),
            np.random.rand(10, 16, 3).astype('float32'),
        ]

        model = FakeModel(rbf, vectors)
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=embedding, y=rand_y, epochs=1)
        preds = model.predict(embedding)

        assert preds[0].shape == (10, 16, 1) and preds[1].shape == (10, 16, 3)

    def test_multiple_conv_layer_model_each_layer_supplied_radial_modified_si_fits_multiple_ro_inputs(self):
        class FakeModel(Model):
            def __init__(self,
                         image,
                         unit_vectors,
                         **kwargs):
                super().__init__(**kwargs)
                self.conv1 = layers.Convolution(image, unit_vectors, radial=Sequential([
                    Dense(64, dynamic=True),
                    Dense(32, dynamic=True),
                    Dense(16, dynamic=True)
                ]), si_units=64, activation='relu')
                self.conv2 = layers.Convolution(image, unit_vectors, radial=Sequential([
                    Dense(64, dynamic=True)
                ]), si_units=32, activation='elu')
                self.conv3 = layers.Convolution(image, unit_vectors, radial=Sequential([
                    Dense(8, dynamic=True),
                    Dense(32, dynamic=True)
                ]), si_units=8, activation=utils.shifted_softplus)

            def call(self, inputs, training=None, mask=None):
                return self.conv3(self.conv2(self.conv1(inputs)))

            def compute_output_shape(self, input_shape):
                return [
                    tf.TensorShape([10, 8, 1]),
                    tf.TensorShape([10, 8, 3])
                ]

        rbf = np.random.rand(10, 10, 80).astype('float32')
        vectors = np.random.rand(10, 10, 3).astype('float32')
        embedding = [
            np.random.rand(10, 16, 1).astype('float32'),
            np.random.rand(10, 16, 3).astype('float32')
        ]
        rand_y = [
            np.random.rand(10, 8, 1).astype('float32'),
            np.random.rand(10, 8, 3).astype('float32'),
        ]

        model = FakeModel(rbf, vectors)
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=embedding, y=rand_y, epochs=1)
        preds = model.predict(embedding)

        assert preds[0].shape == (10, 8, 1) and preds[1].shape == (10, 8, 3)


class TestConvolutionOutputs:

    def test_convolution_multiple_inputs_outputs_correct_tensor_shapes(self):
        rbf = np.random.rand(10, 10, 80).astype('float32')
        vectors = np.random.rand(10, 10, 3).astype('float32')
        embedding = [
            tf.convert_to_tensor(np.random.rand(10, 16, 1).astype('float32')),
            tf.convert_to_tensor(np.random.rand(10, 16, 3).astype('float32'))
        ]
        outputs = layers.Convolution(rbf, vectors).convolution(embedding)
        statements = (
            outputs[0].shape == (10, 16, 1),  # Input 1
            outputs[1].shape == (10, 16, 3),  # Input 1
            outputs[2].shape == (10, 16, 3),  # Input 2
            outputs[3].shape == (10, 16, 1),  # Input 2
            outputs[4].shape == (10, 16, 3)  # Input 2
        )
        assert all(statements)

    def test_concatenation_multiple_inputs_outputs_less_tensors(self):
        rbf = np.random.rand(10, 10, 80).astype('float32')
        vectors = np.random.rand(10, 10, 3).astype('float32')
        embedding = [
            tf.convert_to_tensor(np.random.rand(10, 16, 1).astype('float32')),
            tf.convert_to_tensor(np.random.rand(10, 16, 1).astype('float32')),
            tf.convert_to_tensor(np.random.rand(10, 16, 3).astype('float32')),
            tf.convert_to_tensor(np.random.rand(10, 16, 3).astype('float32'))
        ]
        outputs = layers.Convolution(rbf, vectors).concatenation(embedding)
        assert outputs[0].shape == (10, 32, 1) and outputs[1].shape == (10, 32, 3)


class TestFilter:

    def test_filter_default_radial_outputs_correct_ro2_tensor_shape(self):
        rbf = np.random.rand(10, 10, 80)
        vectors = np.random.rand(10, 10, 3)
        filt = layers.HarmonicFilter(filter_ro=2)([rbf, vectors])

        assert filt.shape == [10, 10, 16, 5]

    def test_filter_provided_conv1d_layer_radial_outputs_correct_ro0_tensor_shape(self):
        rbf = np.random.rand(10, 10, 80)
        vectors = np.random.rand(10, 10, 3)
        layer = Conv1D(filters=16, kernel_size=1)
        filt = layers.HarmonicFilter(radial=layer, filter_ro=0)([rbf, vectors])

        assert filt.shape == [10, 10, 16, 1]

    def test_filter_provided_custom_model_radial_outputs_correct_ro0_tensor_shape(self):
        class FakeModel(Model):
            def call(self, inputs, **kwargs):
                # inputs is of shape (atoms, atoms, rbf)
                w = np.random.rand(inputs.shape[-1], 16)
                return tf.tensordot(inputs, w, axes=[[2], [0]])

        rbf = np.random.rand(10, 10, 80)
        vectors = np.random.rand(10, 10, 3)
        model = FakeModel()  # multiplies rbf by tensor of shape (80, 16)
        filt = layers.HarmonicFilter(radial=model, filter_ro=0)([rbf, vectors])

        assert filt.shape == [10, 10, 16, 1]

    def test_filter_default_radial_gets_trainable_weights(self):
        class FakeModel(MyModel):
            def compute_output_shape(self, input_shape):
                return tf.TensorShape([10, 10, 16, 1])

        rbf = np.random.rand(10, 10, 80)
        vectors = np.random.rand(10, 10, 3)
        rand_y = np.random.rand(10, 10, 16, 1)
        model = FakeModel(layers.HarmonicFilter())
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=[rbf, vectors], y=rand_y, epochs=5)

        assert model.trainable_weights

    def test_filter_ro0_default_radial_fits_random_data(self):
        class FakeModel(MyModel):
            def compute_output_shape(self, input_shape):
                return tf.TensorShape([10, 10, 16, 1])

        rbf = np.random.rand(10, 10, 80)
        vectors = np.random.rand(10, 10, 3)
        rand_y = np.random.rand(10, 10, 16, 1)
        model = FakeModel(layers.HarmonicFilter())
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=[rbf, vectors], y=rand_y, epochs=5)
        pred = model.predict(x=[rbf, vectors])

        assert pred.shape == (10, 10, 16, 1)

    def test_filter_ro1_default_radial_fits_random_data(self):
        class FakeModel(MyModel):
            def compute_output_shape(self, input_shape):
                return tf.TensorShape([10, 10, 16, 3])

        rbf = np.random.rand(10, 10, 80)
        vectors = np.random.rand(10, 10, 3)
        rand_y = np.random.rand(10, 10, 16, 3)
        model = FakeModel(layers.HarmonicFilter(filter_ro=1))
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=[rbf, vectors], y=rand_y, epochs=5)
        pred = model.predict(x=[rbf, vectors])

        assert pred.shape == (10, 10, 16, 3)


class TestSelfInteraction:

    def test_si_multiple_ro_inputs_outputs_correct_tensor_shapes(self):

        inputs = [
            np.random.rand(10, 30, 1),
            np.random.rand(10, 30, 3)
        ]
        si = layers.SelfInteraction(16)
        outputs = si(inputs)

        assert outputs[0].shape == (10, 16, 1) and outputs[1].shape == (10, 16, 3)

    def test_si_multiple_ro_inputs_fits_random_data(self):
        class FakeModel(MyModel):
            def compute_output_shape(self, input_shape):
                return [
                    tf.TensorShape([10, 16, 1]),
                    tf.TensorShape([10, 16, 3])
                ]

        inputs = [
            np.random.rand(10, 30, 1),
            np.random.rand(10, 30, 3)
        ]
        targets = [
            np.random.rand(10, 16, 1),
            np.random.rand(10, 16, 3)
        ]
        model = FakeModel(layers.SelfInteraction(16))
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=inputs, y=targets)
        preds = model.predict(inputs)

        assert preds[0].shape == (10, 16, 1) and preds[1].shape == (10, 16, 3)

    def test_si_multiple_ro_inputs_get_trainable_weights(self):
        class FakeModel(MyModel):
            def compute_output_shape(self, input_shape):
                return [
                    tf.TensorShape([10, 16, 1]),
                    tf.TensorShape([10, 16, 3])
                ]

        inputs = [
            np.random.rand(10, 30, 1),
            np.random.rand(10, 30, 3)
        ]
        targets = [
            np.random.rand(10, 16, 1),
            np.random.rand(10, 16, 3)
        ]
        model = FakeModel(layers.SelfInteraction(16))
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=inputs, y=targets)

        assert model.trainable_weights


class TestEquivariantActivation:

    def test_default_activation_multiple_ro_inputs_outputs_correct_tensor_shapes(self):
        inputs = [
            np.random.rand(10, 16, 1),
            np.random.rand(10, 16, 3)
        ]
        outputs = layers.EquivariantActivation()(inputs)

        assert outputs[0].shape == (10, 16, 1) and outputs[1].shape == (10, 16, 3)

    def test_default_activation_multiple_ro_inputs_get_trainable_weights(self):
        class FakeModel(MyModel):
            def compute_output_shape(self, input_shape):
                return [
                    tf.TensorShape([10, 16, 1]),
                    tf.TensorShape([10, 16, 3])
                ]
        inputs = [
            np.random.rand(10, 16, 1),
            np.random.rand(10, 16, 3)
        ]
        targets = [
            np.random.rand(10, 16, 1),
            np.random.rand(10, 16, 3)
        ]
        model = FakeModel(layers.EquivariantActivation())
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=inputs, y=targets)

        assert model.trainable_weights

    def test_default_activation_multiple_ro_inputs_fits_random_data(self):
        class FakeModel(MyModel):
            def compute_output_shape(self, input_shape):
                return [
                    tf.TensorShape([10, 16, 1]),
                    tf.TensorShape([10, 16, 3])
                ]

        inputs = [
            np.random.rand(10, 16, 1),
            np.random.rand(10, 16, 3)
        ]
        targets = [
            np.random.rand(10, 16, 1),
            np.random.rand(10, 16, 3)
        ]
        model = FakeModel(layers.EquivariantActivation())
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=inputs, y=targets)
        preds = model.predict(inputs)

        assert preds[0].shape == (10, 16, 1) and preds[1].shape == (10, 16, 3)
