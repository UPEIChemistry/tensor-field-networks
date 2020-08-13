from copy import copy

import tensorflow as tf

from tfn.tools.ingredients import builder_ingredient, get_builder
from .job import DefaultJob
from .config_defaults import builder_config


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
            sigma=loader.sigma
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

    def test_fitable(self, fitable, test_data):
        x_test, y_test = test_data
        print(fitable.evaluate(x=x_test, y=y_test, verbose=0))

    def save_results(self, run, fitable):
        path = self.exp_config["run_config"]["model_path"]
        fitable.summary()
        fitable.save(self.exp_config["run_config"]["model_path"])
        run.add_artifact(path)
