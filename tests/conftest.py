import pytest
import numpy as np


@pytest.fixture(scope='session')
def random_data():
    r = np.random.rand(1, 10, 3)
    z = np.random.randint(5, size=(1, 10))
    e = np.random.rand(1,)
    return r, z, e
