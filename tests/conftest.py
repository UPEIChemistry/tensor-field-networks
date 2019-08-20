import pytest
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tfn.layers import Convolution, MolecularConvolution, SelfInteraction, Preprocessing
from tensorflow.python.keras.models import Model


#################
# Data fixtures #
#################

@pytest.fixture(scope='session')
def default_conv_inputs_and_targets():
    z = np.random.randint(5, size=(2, 10, 1))
    r = np.random.rand(2, 10, 3).astype('float32')
    one_hot, rbf, vectors = Preprocessing(5)([r, z])
    inputs = [
        rbf.numpy(),
        vectors.numpy(),
        np.random.rand(2, 10, 16, 1).astype('float32'),
        np.random.rand(2, 10, 8, 3).astype('float32')
    ]
    targets = [
        rbf.numpy(),
        vectors.numpy(),
        np.random.rand(2, 10, 16, 1).astype('float32'),
        np.random.rand(2, 10, 16, 3).astype('float32')
    ]
    return inputs, targets


@pytest.fixture(scope='session')
def molecular_conv_inputs_and_targets():
    r = np.random.rand(2, 10, 3).astype('float32')
    z = np.random.randint(0, 5, size=(2, 10, 1))
    one_hot, rbf, vectors = Preprocessing(5)([r, z])
    inputs = [
        one_hot.numpy(),
        rbf.numpy(),
        vectors.numpy(),
        np.random.rand(2, 10, 16, 1).astype('float32'),
        np.random.rand(2, 10, 8, 3).astype('float32')
    ]
    targets = [
        one_hot.numpy(),
        rbf.numpy(),
        vectors.numpy(),
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


######################
# ConvModel fixtures #
######################

class ConvolutionModel(Model):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
        self.conv1 = Convolution()
        self.conv2 = Convolution()
        self.conv3 = Convolution()

    def call(self, inputs, training=None, mask=None):
        rbf, vectors, *feature_tensors = inputs
        feature_tensors = self.conv1([rbf, vectors] + feature_tensors)
        feature_tensors = self.conv2([rbf, vectors] + feature_tensors)
        return [rbf, vectors] + self.conv3([rbf, vectors] + feature_tensors)

    def compute_output_shape(self, input_shape):
        return [
            tf.TensorShape([shape[0], shape[1], 16, shape[-1]]) for shape in input_shape
        ]


@pytest.fixture
def default_conv_model():
    return ConvolutionModel()


@pytest.fixture
def molecular_conv_model():
    class MolModel(ConvolutionModel):
        def __init__(self,
                     **kwargs):
            super().__init__(**kwargs)
            self.conv1 = MolecularConvolution()
            self.conv2 = MolecularConvolution()
            self.conv3 = MolecularConvolution()

        def call(self, inputs, training=None, mask=None):
            one_hot, rbf, vectors, *features = inputs
            features = self.conv1([one_hot, rbf, vectors] + features)
            features = self.conv2([one_hot, rbf, vectors] + features)
            return [one_hot, rbf, vectors] + self.conv3([one_hot, rbf, vectors] + features)

    return MolModel()


##################
# Model fixtures #
##################

class ScalarModel(Model):
    def __init__(self,
                 max_z=6,
                 gaussian_config=None,
                 radial_factory=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_z = max_z
        self.gaussian_config = gaussian_config
        self.embedding = SelfInteraction(32)
        self.conv1 = MolecularConvolution(radial_factory=radial_factory)
        self.conv2 = MolecularConvolution(radial_factory=radial_factory)
        self.conv3 = MolecularConvolution(radial_factory=radial_factory)

    def call(self, inputs, training=None, mask=None):
        r, z = inputs  # (mols, atoms, 3) and (mols, atoms)
        # Slice r, z for single mol
        one_hot, rbf, vectors = Preprocessing(self.max_z, self.gaussian_config)([r, z])
        embedding = self.embedding(tf.transpose(one_hot, [0, 1, 3, 2]))
        output = self.conv1([one_hot, rbf, vectors] + embedding)
        output = self.conv2([one_hot, rbf, vectors] + output)
        output = self.conv3([one_hot, rbf, vectors] + output)
        return tf.reduce_sum(
            tf.reduce_sum(
                output[0], axis=-2
            ), axis=-2
        )

    def compute_output_shape(self, input_shape):
        mols, atoms, _ = input_shape[0]
        return tf.TensorShape([mols, 1])


class VectorModel(ScalarModel):
    def call(self, inputs, training=None, mask=None):
        r, z = inputs
        one_hot, rbf, vectors = Preprocessing(self.max_z, self.gaussian_config)([r, z])
        embedding = self.embedding(tf.transpose(one_hot, [0, 1, 3, 2]))
        output = self.conv1([one_hot, rbf, vectors] + embedding)
        output = self.conv2([one_hot, rbf, vectors] + output)
        output = self.conv3([one_hot, rbf, vectors] + output)
        return tf.reduce_sum(
            output[1], axis=-2
        )

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape[0])


@pytest.fixture
def scalar_model():
    return ScalarModel()


@pytest.fixture
def vector_model():
    return VectorModel()


@pytest.fixture
def vector_model_no_dummy():
    class NoDummyModel(VectorModel):
        def __init__(self,
                     **kwargs):
            super().__init__(**kwargs)
            self.conv1 = Convolution()
            self.conv2 = Convolution()
            self.conv3 = Convolution()

        def call(self, inputs, training=None, mask=None):
            r, z = inputs
            one_hot, rbf, vectors = Preprocessing(self.max_z, self.gaussian_config)([r, z])
            embedding = self.embedding(tf.transpose(one_hot, [0, 1, 3, 2]))
            output = self.conv1([rbf, vectors] + embedding)
            output = self.conv2([rbf, vectors] + output)
            output = self.conv3([rbf, vectors] + output)
            return tf.reduce_sum(output[1], axis=-2)

    return NoDummyModel()
