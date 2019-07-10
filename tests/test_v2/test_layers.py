import tensorflow as tf
from tfn import layers_v2 as layers
import numpy as np
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv1D


class TestFilter:

    class MyModel(Model):

        def call(self, inputs, **kwargs):
            # inputs is of shape (atoms, atoms, rbf)
            w = np.random.rand(inputs.shape[-1], 16)
            return tf.tensordot(inputs, w, axes=[[2], [0]])

    def test_filter_ro0_default_radial_fits_random_data(self):

        class FakeModel(Model):

            def __init__(self,
                         **kwargs):
                super(FakeModel, self).__init__(**kwargs)
                self.filter = layers.HarmonicFilter()

            def call(self, inputs, training=None, mask=None):
                return self.filter(inputs)

        rbf = np.random.rand(10, 10, 80)
        vectors = np.random.rand(10, 10, 3)
        rand_y = np.random.rand(10, 10, 16, 1)
        model = FakeModel()
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=[rbf, vectors], y=rand_y, epochs=5)
        pred = model.predict(x=[rbf, vectors])

        assert pred.shape == (10, 10, 16, 1)

    def test_filter_ro1_default_radial_fits_random_data(self):

        class FakeModel(Model):

            def __init__(self,
                         **kwargs):
                super(FakeModel, self).__init__(**kwargs)
                self.filter = layers.HarmonicFilter(filter_ro=1)

            def call(self, inputs, training=None, mask=None):
                return self.filter(inputs)

        rbf = np.random.rand(10, 10, 80)
        vectors = np.random.rand(10, 10, 3)
        rand_y = np.random.rand(10, 10, 16, 3)
        model = FakeModel()
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=[rbf, vectors], y=rand_y, epochs=5)
        pred = model.predict(x=[rbf, vectors])

        assert pred.shape == (10, 10, 16, 3)

    def test_filter_default_radial_outputs_ro0(self):
        rbf = np.random.rand(10, 10, 80)
        vectors = np.random.rand(10, 10, 3)
        filt = layers.HarmonicFilter()([rbf, vectors])

        assert filt.shape == [10, 10, 16, 1]

    def test_filter_default_radial_outputs_ro1(self):
        rbf = np.random.rand(10, 10, 80)
        vectors = np.random.rand(10, 10, 3)
        filt = layers.HarmonicFilter(filter_ro=1)([rbf, vectors])

        assert filt.shape == [10, 10, 16, 3]

    def test_filter_default_radial_outputs_ro2(self):
        rbf = np.random.rand(10, 10, 80)
        vectors = np.random.rand(10, 10, 3)
        filt = layers.HarmonicFilter(filter_ro=2)([rbf, vectors])

        assert filt.shape == [10, 10, 16, 5]

    def test_filter_provided_custom_model_radial_outputs_ro0(self):
        rbf = np.random.rand(10, 10, 80)
        vectors = np.random.rand(10, 10, 3)
        model = self.MyModel()  # multiplies rbf by tensor of shape (80, 16)
        filt = layers.HarmonicFilter(radial=model, filter_ro=0)([rbf, vectors])

        assert filt.shape == [10, 10, 16, 1]

    def test_filter_provided_conv1d_layer_radial_outputs_ro0(self):
        rbf = np.random.rand(10, 10, 80)
        vectors = np.random.rand(10, 10, 3)
        layer = Conv1D(filters=16, kernel_size=1)
        filt = layers.HarmonicFilter(radial=layer, filter_ro=0)([rbf, vectors])

        assert filt.shape == [10, 10, 16, 1]


class TestConvolution:

    def test_filter_default_radial_fits_single_random_ro0_data(self):

        class FakeModel(Model):

            def __init__(self,
                         image,
                         unit_vectors,
                         **kwargs):
                super(FakeModel, self).__init__(**kwargs)
                self.conv = layers.Convolution(image, unit_vectors)

            def call(self, inputs, training=None, mask=None):
                return self.conv(inputs)

        rbf = np.random.rand(10, 10, 80).astype('float32')
        vectors = np.random.rand(10, 10, 3).astype('float32')
        embedding = np.random.rand(10, 16, 1).astype('float32')
        rand_y = [
            np.random.rand(10, 16, 1).astype('float32'),
            np.random.rand(10, 16, 1).astype('float32')
        ]
        model = FakeModel(rbf, vectors)
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)

        model.fit(x=embedding, y=rand_y)
        pred = model.predict(x=embedding)

        assert pred[0].shape == (10, 16, 1) and pred[1].shape == (10, 16, 3)

    def test_filter_default_radial_fits_single_random_ro1_data(self):

        class FakeModel(Model):

            def __init__(self,
                         image,
                         unit_vectors,
                         **kwargs):
                super(FakeModel, self).__init__(**kwargs)
                self.conv = layers.Convolution(image, unit_vectors)

            def call(self, inputs, training=None, mask=None):
                return self.conv(inputs)

        rbf = np.random.rand(10, 10, 80).astype('float32')
        vectors = np.random.rand(10, 10, 3).astype('float32')
        embedding = np.random.rand(10, 16, 3).astype('float32')
        rand_y = [
            np.random.rand(10, 16, 3).astype('float32'),
            np.random.rand(10, 16, 1).astype('float32'),
            np.random.rand(10, 16, 3).astype('float32')
        ]
        model = FakeModel(rbf, vectors)
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)

        model.fit(x=embedding, y=rand_y)
        pred = model.predict(x=embedding)

        assert pred[0].shape == (10, 16, 3) and pred[1].shape == (10, 16, 1) and pred[2].shape == (10, 16, 3)

    def test_filter_default_radial_fits_multiple_ro_random_data(self):

        class FakeModel(Model):

            def __init__(self,
                         image,
                         unit_vectors,
                         **kwargs):
                super(FakeModel, self).__init__(**kwargs)
                self.conv = layers.Convolution(image, unit_vectors)

            def call(self, inputs, training=None, mask=None):
                return self.conv(inputs)

        rbf = np.random.rand(10, 10, 80).astype('float32')
        vectors = np.random.rand(10, 10, 3).astype('float32')
        embedding = [
            np.random.rand(10, 16, 1).astype('float32'),
            np.random.rand(10, 16, 3).astype('float32'),

        ]
        rand_y = [
            np.random.rand(10, 16, 1).astype('float32'),
            np.random.rand(10, 16, 3).astype('float32'),
            np.random.rand(10, 16, 3).astype('float32'),
            np.random.rand(10, 16, 1).astype('float32'),
            np.random.rand(10, 16, 3).astype('float32')
        ]
        model = FakeModel(rbf, vectors)
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)

        model.fit(x=embedding, y=rand_y)
        pred = model.predict(x=embedding)

        statements = (
            pred[0].shape == (10, 16, 1),
            pred[1].shape == (10, 16, 3),
            pred[2].shape == (10, 16, 3),
            pred[3].shape == (10, 16, 1),
            pred[4].shape == (10, 16, 3)
        )

        for statement in statements:
            assert statement
