import os
from copy import copy
from pathlib import Path

import tensorflow as tf
import numpy as np

from tfn.tools.ingredients import builder_ingredient, get_builder
from .job import DefaultJob
from .config_defaults import builder_config
from ..converters import ndarrays_to_xyz


class SingleModel(DefaultJob):
    @property
    def config_defaults(self):
        config = copy(super().config_defaults)
        config["ingredients"].extend([builder_ingredient])
        config["builder_config"] = copy(builder_config)
        return config

    def load_fitable(self, loader, fitable_config: dict = None):
        fitable_config = fitable_config or self.exp_config["builder_config"]
        conf = dict(
            **fitable_config,
            max_z=loader.max_z,
            num_points=loader.num_points,
            mu=loader.mu,
            sigma=loader.sigma,
        )
        builder = get_builder(**conf)
        run_config = self.exp_config["run_config"]
        compile_kwargs = dict(
            loss=run_config["loss"],
            loss_weights=run_config["loss_weights"],
            optimizer=tf.keras.optimizers.Adam(**run_config["optimizer_kwargs"]),
            metrics=run_config["metrics"],
            run_eagerly=run_config["run_eagerly"],
        )
        if run_config["use_strategy"]:
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                model = builder.get_model()
                model.compile(**compile_kwargs)
        else:
            model = builder.get_model()
            model.compile(**compile_kwargs)
        return model

    def test_fitable(self, fitable, test_data, run=None):
        x_test, y_test = test_data
        print(f"Test split results: {fitable.evaluate(x=x_test, y=y_test, verbose=0)}")
        if self.exp_config["run_config"]["write_test_results"]:
            self.write_test_results(fitable, run, x_test, y_test)

    def write_test_results(self, fitable, run, x_test, y_test):
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

    def save_model(self, run, fitable):
        path = self.exp_config["run_config"]["model_path"]
        fitable.summary()
        fitable.save(self.exp_config["run_config"]["model_path"])
        run.add_artifact(path)
