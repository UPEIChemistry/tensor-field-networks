import tensorflow as tf
from tfn import layers
import numpy as np
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, Conv2D


class TestHarmonicFilterRotationOrders:
    def test_ro0_filter(self, random_rbf_and_vectors):
        rbf, vectors = random_rbf_and_vectors
        output = layers.HarmonicFilter(
            radial=Dense(16, dynamic=True),
            filter_ro=0
        )([rbf, vectors])
        assert output.shape[-1] == 1

    def test_ro1_filter(self, random_rbf_and_vectors):
        rbf, vectors = random_rbf_and_vectors
        output = layers.HarmonicFilter(
            radial=Dense(16, dynamic=True),
            filter_ro=1
        )([rbf, vectors])
        assert output.shape[-1] == 3

    def test_ro2_filter(self, random_rbf_and_vectors):
        rbf, vectors = random_rbf_and_vectors
        output = layers.HarmonicFilter(
            radial=Dense(16, dynamic=True),
            filter_ro=2
        )([rbf, vectors])
        assert output.shape[-1] == 5


class TestHarmonicFilterVariousRadials:
    def test_dense_radial_correct_output_shape(self, random_rbf_and_vectors):
        rbf, vectors = random_rbf_and_vectors
        output = layers.HarmonicFilter(radial=Dense(16, dynamic=True), filter_ro=0)([rbf, vectors])
        assert output.shape == (2, 10, 10, 16, 1)

    def test_conv_radial_correct_output_shape(self, random_rbf_and_vectors):
        rbf, vectors = random_rbf_and_vectors
        output = layers.HarmonicFilter(radial=Conv2D(16, 1, dynamic=True), filter_ro=0)([rbf, vectors])
        assert output.shape == (2, 10, 10, 16, 1)

    def test_model_radial_correct_output_shape(self, random_rbf_and_vectors):
        rbf, vectors = random_rbf_and_vectors
        radial = Sequential([Dense(32, dynamic=True), Dense(16, dynamic=True)])
        output = layers.HarmonicFilter(radial=radial, filter_ro=0)([rbf, vectors])
        assert output.shape == (2, 10, 10, 16, 1)


class TestHarmonicFilterTrainableWeights:
    def test_dense_radial_correct_num_trainable_weights(self, random_rbf_and_vectors):
        rbf, vectors = random_rbf_and_vectors
        target = np.random.rand(2, 10, 10, 16, 1)

        class HFModel(Model):
            def __init__(self):
                super().__init__()
                self.filter = layers.HarmonicFilter(radial=Dense(16, dynamic=True), filter_ro=0)

            def call(self, inputs, training=None, mask=None):
                return self.filter(inputs)

            def compute_output_shape(self, input_shape):
                return tf.TensorShape([2, 10, 10, 16, 1])

        model = HFModel()
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=[rbf, vectors], y=target)
        assert len(model.trainable_weights) == 2


class TestSelfInteraction:
    class SIModel(Model):
        def __init__(self,
                     activity_regularizer=None,
                     **kwargs):
            super().__init__(**kwargs)
            self.si = layers.SelfInteraction(32, activity_regularizer=activity_regularizer or None)

        def call(self, tensors, training=None, mask=None):
            return self.si(tensors)

        def compute_output_shape(self, input_shape):
            return [tf.TensorShape([shape[0], shape[1], 32, shape[-1]]) for shape in input_shape]

    def test_correct_output_shapes(self, random_features_and_targets):
        inputs, targets = random_features_and_targets
        si = layers.SelfInteraction(32)
        outputs = si(inputs)
        assert outputs[0].shape == (2, 10, 32, 1) and outputs[1].shape == (2, 10, 32, 3)

    def test_correct_num_trainable_weights(self, random_features_and_targets):
        inputs, targets = random_features_and_targets
        targets = np.random.rand(2, 10, 32, 1), np.random.rand(2, 10, 32, 3)

        model = self.SIModel()
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=inputs, y=targets)

        assert len(model.trainable_weights) == 2

    def test_regularization(self, random_features_and_targets):
        inputs, targets = random_features_and_targets
        targets = np.random.rand(2, 10, 32, 1), np.random.rand(2, 10, 32, 3)
        model = self.SIModel(activity_regularizer=tf.keras.regularizers.l2(0.01))
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=inputs, y=targets)


class TestEquivariantActivation:
    class ActModel(Model):
        def __init__(self,
                     activity_regularizer=None,
                     **kwargs):
            super().__init__(**kwargs)
            self.a = layers.EquivariantActivation(activity_regularizer=activity_regularizer or None)

        def call(self, tensors, training=None, mask=None):
            return self.a(tensors)

        def compute_output_shape(self, input_shape):
            return [tf.TensorShape([*shape]) for shape in input_shape]

    def test_correct_output_shapes(self, random_features_and_targets):
        inputs, targets = random_features_and_targets
        a = layers.EquivariantActivation()
        outputs = a(inputs)
        assert all([i.shape == o.shape for i, o in zip(inputs, outputs)])

    def test_correct_num_trainable_weights(self, random_features_and_targets):
        inputs, targets = random_features_and_targets
        targets = [np.random.rand(*i.shape) for i in inputs]
        model = self.ActModel()
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=inputs, y=targets)

        assert len(model.trainable_weights) == 2

    def test_regularization(self, random_features_and_targets):
        inputs, targets = random_features_and_targets
        targets = [np.random.rand(*i.shape) for i in inputs]
        model = self.ActModel(activity_regularizer=tf.keras.regularizers.l2(0.01))
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=inputs, y=targets)