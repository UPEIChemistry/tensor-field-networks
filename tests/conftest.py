import pytest
import numpy as np
import tensorflow as tf
from tfn.layers import Convolution
from tensorflow.python.keras.models import Model


@pytest.fixture(scope='session')
def random_inputs_and_targets():
    rbf = np.random.rand(2, 10, 10, 80).astype('float32')
    vectors = np.random.rand(2, 10, 10, 3).astype('float32')
    inputs = [
        rbf,
        vectors,
        np.random.rand(2, 10, 16, 1).astype('float32'),
        np.random.rand(2, 10, 8, 3).astype('float32')
    ]
    targets = [
        rbf,
        vectors,
        np.random.rand(2, 10, 16, 1).astype('float32'),
        np.random.rand(2, 10, 16, 3).astype('float32')
    ]
    return inputs, targets


@pytest.fixture(scope='session')
def random_features_and_targets():
    features = [
        np.random.rand(2, 10, 16, 1).astype('float32'),
        np.random.rand(2, 10, 8, 3).astype('float32')
    ]
    targets = [
        np.random.rand(2, 10, 16, 1).astype('float32'),
        np.random.rand(2, 10, 16, 3).astype('float32')
    ]
    return features, targets


@pytest.fixture(scope='session')
def random_rbf_and_vectors():
    rbf = np.random.rand(2, 10, 10, 80).astype('float32')
    vectors = np.random.rand(2, 10, 10, 3).astype('float32')
    return rbf, vectors


@pytest.fixture(scope='session')
def random_cartesians_and_z():
    z = np.random.randint(5, size=(2, 10, 1))
    r = np.random.rand(2, 10, 3).astype('float32')
    return r, z


@pytest.fixture
def default_conv_model():
    class MultiLayerConvModel(Model):
        def __init__(self,
                     **kwargs):
            super().__init__(**kwargs)
            self.conv1 = Convolution()
            self.conv2 = Convolution()
            self.conv3 = Convolution()

        def call(self, inputs, training=None, mask=None):
            rbf, vectors, feature_tensors = inputs[0:1], inputs[1:2], inputs[2:]
            feature_tensors = self.conv1(rbf + vectors + feature_tensors)
            feature_tensors = self.conv2(rbf + vectors + feature_tensors)
            return rbf + vectors + self.conv3(rbf + vectors + feature_tensors)

        def compute_output_shape(self, input_shape):
            if not isinstance(input_shape, list):
                input_shape = [input_shape]
            return [
                tf.TensorShape([shape[0], shape[1], 16, shape[-1]]) for shape in input_shape
            ]
    return MultiLayerConvModel()

