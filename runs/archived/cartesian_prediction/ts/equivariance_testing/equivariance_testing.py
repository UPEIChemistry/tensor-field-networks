from pathlib import Path

import numpy as np

from tfn.layers.utils import rotation_matrix
from tfn.tools.builders import CartesianBuilder
from tfn.tools.callbacks import CartesianMetrics
from tfn.tools.loaders import TSLoader

from tensorflow.keras.callbacks import TensorBoard

np.random.seed(0)
loader = TSLoader(
    "/home/riley/dev/python/data/ts.hdf5",
    splitting="custom",
    map_points=False,
    pre_load=False,
)
train, val, _ = loader.load_data(cache=False, remove_noise=True, shuffle=False)
(z, r, p), (ts,) = val
model = CartesianBuilder(loader.max_z, loader.num_points,).get_model()
model.compile(optimizer="adam", loss="mae")
path = Path("storage")
cartesian_writer = CartesianMetrics(path / "cartesians", train, val)
model.fit(
    *train,
    validation_data=((z, r, p), (ts,)),
    epochs=500,
    callbacks=[TensorBoard(path / "logs"), cartesian_writer],
    verbose=2,
)
rotation = rotation_matrix()  # 90 degrees around x-axis
r_rotated, p_rotated, ts_rotated = [np.dot(a, rotation) for a in (r, p, ts)]
cartesian_writer.write_cartesians(
    [(z, r_rotated, p_rotated), (ts,)], path / "rotated_structures"
)

rotation_results = [
    np.mean(np.abs(model.predict((z, r_rotated, p_rotated), verbose=0) - ts_rotated)),
    np.mean(np.abs(model.predict((z, r, p), verbose=0) - ts)),
]
commutative_results = [
    np.mean(np.abs(model.predict((z, p, r), verbose=0) - ts)),
    np.mean(np.abs(model.predict((z, r, p), verbose=0) - ts)),
]

print(
    f"{round(rotation_results[0], 4)} vs. {round(rotation_results[1], 4)} for "
    f"rotated/non-rotated model evaluation"
)

print(
    f"{round(commutative_results[0], 4)} vs. {round(commutative_results[1], 4)} for "
    f"swapped input model evaluation"
)

assert np.isclose(*rotation_results, atol=0.1)
assert np.isclose(*commutative_results, atol=0.1)
