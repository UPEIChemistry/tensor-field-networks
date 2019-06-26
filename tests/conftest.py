import pytest
import numpy as np


@pytest.fixture(scope='session')
def random_data():
    r = np.random.rand(1, 10, 3)
    z = np.random.randint(10, size=(1, 10))
    e = np.random.randn()
    return r, z, e
