import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.keras.models import Model

from tfn.layers import Convolution, MolecularConvolution, Preprocessing, SelfInteraction


#################
# Data fixtures #
#################

@pytest.fixture(scope='session')
def random_onehot_rbf_vectors():
    one_hot = np.random.rand(2, 10, 5)
    rbf = np.random.rand(2, 10, 10, 80).astype('float32')
    vectors = np.random.rand(2, 10, 10, 3).astype('float32')
    return one_hot, rbf, vectors


@pytest.fixture(scope='session')
def random_cartesians_and_z():
    z = np.random.randint(5, size=(2, 10, 1))
    r = np.random.rand(2, 10, 3).astype('float32')
    return r, z


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
