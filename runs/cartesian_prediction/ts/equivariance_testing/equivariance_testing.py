from pathlib import Path

import numpy as np

from tfn.utils import rotation_matrix
from tfn.tools.builders import CartesianBuilder
from tfn.tools.callbacks import CartesianMetrics
from tfn.tools.loaders import TSLoader

from tensorflow.keras.callbacks import TensorBoard

np.random.seed(0)
loader = TSLoader(
    "/home/riley/dev/python/data/ts.hdf5", splitting="90:10:0", map_points=False
)
train, ((z, r, p), (ts,)), _ = loader.load_data(cache=True, remove_noise=True)
model = CartesianBuilder(loader.max_z, loader.num_points,).get_model()
model.compile(optimizer="adam", loss="mae", run_eagerly=True)
path = Path("storage")
cartesian_writer = CartesianMetrics(path / "cartesians", *loader.load_data())
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
print(
    f"{round(rotation_results[0], 4)} vs. {round(rotation_results[1], 4)} for "
    f"rotated/non-rotated model evaluation"
)
assert np.isclose(*rotation_results, atol=0.1)
