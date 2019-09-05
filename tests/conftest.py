import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K

from tfn.layers import MolecularConvolution, Preprocessing, SelfInteraction


# ===== Parser ===== #

def pytest_addoption(parser):
    parser.addoption(
        '--eager', action='store_true', default=False
    )
    parser.addoption(
        '--dynamic', action='store_true', default=False
    )


@pytest.fixture(scope='session')
def dynamic(request):
    return request.config.getoption('--dynamic')


@pytest.fixture(scope='session')
def eager(request):
    return request.config.getoption('--eager')


# ===== Data Fixtures ===== #

@pytest.fixture(scope='session')
def random_onehot_rbf_vectors():
    one_hot = np.random.randint(0, 2, size=[2, 10, 5])
    rbf = np.random.rand(2, 10, 10, 80).astype('float32')
    vectors = np.random.rand(2, 10, 10, 3).astype('float32')
    return one_hot, rbf, vectors


@pytest.fixture(scope='session')
def random_cartesians_and_z():
    r = np.random.rand(2, 10, 3).astype('float32')
    z = np.random.randint(5, size=(2, 10, 1))
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
