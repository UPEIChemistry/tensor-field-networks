import pytest
import numpy as np


@pytest.fixture(scope='session')
def random_data():
    r = np.random.rand(10, 3)
    z = np.random.randint(5, size=(5, ))
    e = np.random.randn()
    return r, z, e
