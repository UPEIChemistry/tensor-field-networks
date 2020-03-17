import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K

from tfn.layers import MolecularConvolution, Preprocessing, SelfInteraction


# ===== Parser ===== #

def pytest_addoption(parser):
    parser.addoption(
        '--not-eager', action='store_false', default=True
    )
    parser.addoption(
        '--not-dynamic', action='store_false', default=True
    )


@pytest.fixture(scope='session')
def dynamic(request):
    return request.config.getoption('--not-dynamic')


@pytest.fixture(scope='session')
def eager(request):
    return request.config.getoption('--not-eager')


# ===== Data Fixtures ===== #

@pytest.fixture(scope='session')
def random_onehot_rbf_vectors():
    one_hot = np.random.randint(0, 2, size=[2, 10, 6])
    rbf = np.random.rand(2, 10, 10, 80).astype('float32')
    vectors = np.random.rand(2, 10, 10, 3).astype('float32')
    return one_hot, rbf, vectors


@pytest.fixture(scope='session')
def random_cartesians_and_z():
    r = np.random.rand(2, 10, 3).astype('float32')
    z = np.random.randint(6, size=(2, 10))
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
