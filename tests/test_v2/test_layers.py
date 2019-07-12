import tensorflow as tf
from tfn import layers
import numpy as np
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv1D


class MyModel(Model):

    def __init__(self,
                 layer,
                 **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.layer = layer

    def call(self, inputs, training=None, mask=None):
        return self.layer(inputs)


class TestConvolution:

    def test_convolutoin_fits_cartesian_and_ro0_embedding_input(self):
        pass

    def test_convolution_multiple_inputs_outputs_tensors(self):
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

    def test_filter_default_radial_outputs_ro2(self):
        rbf = np.random.rand(10, 10, 80)
        vectors = np.random.rand(10, 10, 3)
        filt = layers.HarmonicFilter(filter_ro=2)([rbf, vectors])

        assert filt.shape == [10, 10, 16, 5]

    def test_filter_provided_conv1d_layer_radial_outputs_ro0(self):
        rbf = np.random.rand(10, 10, 80)
        vectors = np.random.rand(10, 10, 3)
        layer = Conv1D(filters=16, kernel_size=1)
        filt = layers.HarmonicFilter(radial=layer, filter_ro=0)([rbf, vectors])

        assert filt.shape == [10, 10, 16, 1]

    def test_filter_provided_custom_model_radial_outputs_ro0(self):
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
        model = FakeModel(layers.HarmonicFilter())
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=[rbf, vectors], y=rand_y, epochs=5)
        pred = model.predict(x=[rbf, vectors])

        assert pred.shape == (10, 10, 16, 3)


class TestSelfInteraction:

    def test_si_multiple_ro_inputs_call_produces_correct_output_tensors(self):

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
