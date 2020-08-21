import numpy as np
import pytest


# ===== Parser ===== #


def pytest_addoption(parser):
    parser.addoption("--not-eager", action="store_false", default=True)
    parser.addoption("--not-dynamic", action="store_false", default=True)


@pytest.fixture(scope="session")
def dynamic(request):
    return request.config.getoption("--not-dynamic")


@pytest.fixture(scope="session")
def eager(request):
    return request.config.getoption("--not-eager")


# ===== Data Fixtures ===== #


@pytest.fixture(scope="session")
def random_onehot_rbf_vectors():
    one_hot = np.random.randint(0, 2, size=[2, 1, 6])
    rbf = np.random.rand(2, 1, 1, 80).astype("float32")
    vectors = np.random.rand(2, 1, 1, 3).astype("float32")
    return one_hot, rbf, vectors


@pytest.fixture(scope="session")
def random_z_and_cartesians():
    r = np.random.rand(2, 1, 3).astype("float32")
    z = np.random.randint(6, size=(2, 1))
    return z, r


@pytest.fixture(scope="session")
def random_features_and_targets():
    features = [
        np.random.rand(2, 1, 16, 1).astype("float32"),
        np.random.rand(2, 1, 8, 3).astype("float32"),
    ]
    targets = [
        np.random.rand(2, 1, 16, 1).astype("float32"),
        np.random.rand(2, 1, 16, 3).astype("float32"),
    ]
    return features, targets
