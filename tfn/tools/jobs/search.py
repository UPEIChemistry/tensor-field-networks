from copy import copy
from pathlib import Path
from typing import Union, List

from kerastuner import HyperParameters, Hyperband, RandomSearch, Tuner
from sacred.run import Run
from sklearn.model_selection import ParameterGrid
import tensorflow as tf
from tensorflow.keras.models import Model

from . import DefaultJob, SingleModel
from .config_defaults import factory_config, tuner_config
from ..ingredients import get_hyper_factory, hyper_factory_ingredient


class Search(DefaultJob):
    def __init__(self, *args, tuner="random", **kwargs):
        """
        :param tuner: str. Possible options include: 'random', 'hyperband'
        """
        if tuner == "random":
            self.tuner: type(Tuner) = RandomSearch
        elif tuner == "hyperband":
            self.tuner: type(Tuner) = Hyperband
        else:
            raise ValueError("Unknown value: `{}` for parameter `tuner`".format(tuner))
        super().__init__(*args, **kwargs)

    @property
    def config_defaults(self):
        config = copy(super().config_defaults)
        config["ingredients"].append(hyper_factory_ingredient)
        config["factory_config"] = copy(factory_config)
        config["tuner_config"] = copy(tuner_config)
        config["loader_config"]["splitting"] = "70:20:10"
        if self.tuner is RandomSearch:
            config["tuner_config"]["max_trials"] = 100
        else:
            config["tuner_config"]["max_epochs"] = 30
        return config

    def resolve_space(self):
        hp = HyperParameters()
        for name, conf in self.exp_config["search_space"].items():
            space_type = conf["type"].lower()
            kwargs = conf["kwargs"]
            kwargs["name"] = name
            if space_type == "choice":
                hp.Choice(**kwargs)
            elif space_type == "float":
                hp.Float(**kwargs)
            elif space_type == "int":
                hp.Int(**kwargs)
            elif space_type == "boolean":
                hp.Boolean(**kwargs)
            elif space_type == "fixed":
                hp.Fixed(**kwargs)

        return hp

    def load_fitable(self, loader, fitable_config: dict = None):
        hp = self.resolve_space()
        fitable_config = fitable_config or self.exp_config["factory_config"]
        conf = dict(
            **fitable_config,
            max_z=loader.max_z,
            num_points=loader.num_points,
            mu=loader.mu,
            sigma=loader.sigma
        )
        tuner_kwargs = dict(
            hypermodel=get_hyper_factory(**conf),
            hyperparameters=hp,
            tune_new_entries=False,
            **self.exp_config["tuner_config"]
        )
        if self.exp_config["run_config"]["use_strategy"]:
            tuner_kwargs["distribution_strategy"] = tf.distribute.MirroredStrategy()
        tuner = self.tuner(**tuner_kwargs)
        return tuner

    def test_fitable(self, fitable: Union[Model, Tuner], test_data: tuple):
        x_test, y_test = test_data
        models: List[Model] = fitable.get_best_models(
            self.exp_config["tuner_config"]["num_models_to_test"]
        )
        for model in models:
            model.evaluate(x_test, y_test, verbose=0)

    def save_results(self, run: Run, fitable: Union[Model, Tuner]):
        fitable.results_summary()
        for i, model in enumerate(
            fitable.get_best_models(self.exp_config["run_config"]["num_models_to_test"])
        ):
            path = Path(
                self.exp_config["run_config"]["model_path"]
            ).parent / "model_{}.h5".format(i)
            model.save(path)


class GridSearch(SingleModel):
    def __init__(self, *args, **kwargs):
        self.total_models = kwargs.pop("total_models", None)
        super().__init__(*args, **kwargs)

    def main(self, run, fitable=None, loader_config=None, fitable_config=None):
        config_grid = ParameterGrid(self.exp_config["grid_config"])
        for i, config in enumerate(config_grid):
            if (i + 1) == self.total_models:  # Stop when hitting max models
                break
            print("\nConfig set (not showing defaults): {}".format(config))
            [
                config.setdefault(k, v)
                for k, v in self.exp_config["builder_config"].items()
            ]
            self.new_model_path(i)
            try:
                super().main(run, fitable_config=config)
            except Exception as e:
                print(
                    "Encountered exception in search, skipping configuration...\n Error "
                    "message: {}".format(e)
                )
                pass
