import numpy as np
from pathlib import Path
from sacred.run import Run
from tensorflow.keras.models import Model

from . import KerasJob
from ..converters import ndarrays_to_xyz


class Regression(KerasJob):
    pass


class StructurePrediction(Regression):
    def _test_fitable(self, run: Run, fitable: Model, test_data: tuple):
        """
        Write cartesian

        :param fitable:
        :param run:
        :param test_data:
        :return:
        """
        x_test, y_test = test_data
        y_pred = fitable.predict(x_test)
        logdir = Path(run.observers[0].dir).absolute() / "cartesians"
        for i, (z, r, p, ts_pred, ts_true) in enumerate(
            zip(x_test[0], x_test[1], x_test[2], y_pred, y_test[0])
        ):
            loss = fitable.evaluate(
                x=[np.expand_dims(a, axis=0) for a in (z, r, p)],
                y=np.expand_dims(ts_true, axis=0),
                verbose=0,
            )
            ndarrays_to_xyz(
                ts_pred,
                z,
                logdir / "predicted" / f"{i}_pred.xyz",
                message=f"loss value: {loss}",
            )
            ndarrays_to_xyz(ts_true, z, logdir / "true" / f"{i}_true.xyz")
            ndarrays_to_xyz(r, z, logdir / "reactant" / f"{i}_reactant.xyz")
            ndarrays_to_xyz(p, z, logdir / "product" / f"{i}_product.xyz")
