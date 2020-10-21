from tfn.tools.loaders import TSLoader
import numpy as np


def get_atomic_histogram_data(z):
    z = np.where(z == 0, np.nan, z)
    z = np.where(z == 1, np.nan, z)
    return [np.count_nonzero(z == i) for i in range(36)]


loader = TSLoader(
    path="/home/riley/dev/python/data/ts.hdf5", splitting=None, map_points=False
)
x, _ = loader.load_data(remove_noise=True, shuffle=False)[0]
z, *_ = x

print(f"Size data: {np.count_nonzero(np.where(z == 1, 0, z), axis=1)}")
print(f"Type data: {get_atomic_histogram_data(z)}")
