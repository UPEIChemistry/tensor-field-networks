from copy import copy
import socket
from pathlib import Path
from typing import List, Union, Tuple

from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver, RunObserver
from sacred.run import Run
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard
from tensorflow.keras.models import Model
from kerastuner.engine.tuner import Tuner

from ..callbacks import ClassificationMetrics
from .config_defaults import run_config, loader_config, tb_config, lr_config
from ..ingredients import data_ingredient, get_data_loader
from ..loaders import DataLoader


class Job(object):
    def __init__(
        self,
        exp_config: dict = None,
        add_defaults: bool = True,
        mongo_hostnames: list = None,
    ):
        exp_config = exp_config or dict()
        if add_defaults:
            self.exp_config = self.add_config_defaults(exp_config)
        else:
            self.exp_config = exp_config
        if mongo_hostnames is None:
            mongo_hostnames = ["tater"]
        self.mongo_hostnames = mongo_hostnames

        self._experiment = None
        self._observers = []

    @staticmethod
    def set_config_defaults(d: dict, values: dict):
        for k, v in values.items():
            d.setdefault(k, v)

    def add_config_defaults(self, ec: dict):
        for name, conf in self.config_defaults.items():
            if name in ec:
                self.set_config_defaults(ec[name], conf)
            else:
                ec.setdefault(name, conf)
        return ec

    @property
    def default_observers(self):
        observers = []
        if socket.gethostname() in self.mongo_hostnames:
            observers.append(
                MongoObserver(
                    url=f"mongodb://sample:password@localhost:27017/?authMechanism=SCRAM-SHA-1",
                    db_name="db",
                )
            )
        observers.append(
            FileStorageObserver(self.exp_config.get("storage_dir", "./sacred_storage"))
        )
        return observers

    def update_observers(self, o: List[RunObserver]):
        """
        ONLY USE BEFORE CALLING `self.experiment` AS OBSERVERS CANNOT BE SET AFTER THE EXPERIMENT
        IS CREATED.

        :param o: List of sacred RunObservers to update Job observers.
        """
        self._observers.extend(o)

    def override_observers(self, o: List[RunObserver]):
        """
        ONLY USE BEFORE CALLING `self.experiment`. Replace defaults with new list of
        RunObserver objects.
        :param o: List of new sacred RunObservers
        """
        self._observers = o

    @property
    def experiment(self):
        if self._experiment is None:
            self._experiment = Experiment(
                name=self.exp_config.get("name"),
                ingredients=self.exp_config.get("ingredients"),
            )
            observers = self._observers or self.default_observers
            self._experiment.observers.extend(observers)
            self._experiment.add_config(self.exp_config)
            if not self.exp_config["run_config"]["capture_output"]:
                self._experiment.captured_out_filter = (
                    lambda *args, **kwargs: "Output capturing turned off."
                )
        return self._experiment

    @property
    def config_defaults(self):
        """
        Defines default values for the various config dictionaries required for the Job.

        :return: dict. Experiment dictionary containing necessary config(s) for the Job.
        """
        raise NotImplementedError

    def run(self):
        @self.experiment.main
        def main(_run):
            self.main(_run)

        self.experiment.run()

    def main(
        self,
        run,
        fitable: Union[Model, Tuner] = None,
        loader_config: dict = None,
        fitable_config: dict = None,
    ):
        raise NotImplementedError

    def load_data(self, config: dict = None) -> Tuple[DataLoader, Tuple]:
        """
        Obtains a loader using ingredients.get_loader and self.exp_config['loader_config']

        :return: Loader object and the data returned by that Loader's get_data method.
        """
        raise NotImplementedError

    def load_fitable(
        self, loader: DataLoader, fitable_config: dict = None
    ) -> Union[Model, Tuner]:
        """
        Defines and compiles a fitable (keras.model or keras_tuner.tuner) which implements
        a 'fit' method. This method calls either get_builder, or get_hyper_factory, depending on
        which type of fitable is beind loaded.

        :return: Model or Tuner object.
        """
        raise NotImplementedError

    def fit(self, fitable: Union[Model, Tuner], data: tuple) -> Union[Model, Tuner]:
        """
        Fits a provided fitable to some provided data.

        :return: Model or Tuner object fitted to data.
        """
        raise NotImplementedError

    def test_fitable(self, fitable: Union[Model, Tuner], test_data: tuple) -> float:
        """
        :param fitable: Model or Tuner.
        :param test_data: tuple. contains x_test & y_test
        :return: float. Scalar test_loss value.
        """
        raise NotImplementedError

    def save_results(self, run: Run, fitable: Union[Model, Tuner]):
        raise NotImplementedError


class DefaultJob(Job):
    @property
    def config_defaults(self):
        return {
            "ingredients": [data_ingredient],
            "run_config": copy(run_config),
            "loader_config": copy(loader_config),
            "tb_config": copy(tb_config),
            "lr_config": copy(lr_config),
        }

    def main(self, run, fitable=None, loader_config=None, fitable_config=None):
        loader, data = self.load_data(loader_config)
        fitable = fitable or self.load_fitable(loader, fitable_config)
        fitable = self.fit(fitable, data[:-1])
        if self.exp_config["run_config"]["test"]:
            self.test_fitable(fitable, data[-1])
        if self.exp_config["run_config"]["save_model"]:
            self.save_results(run, fitable)
        return fitable

    def load_data(self, config: dict = None) -> Tuple[DataLoader, Tuple]:
        config = config or self.exp_config["loader_config"]
        loader = get_data_loader(**config)
        if self.exp_config["run_config"]["select_few"]:
            (x_train, y_train), val, (x_test, y_test) = loader.few_examples(
                **config["load_kwargs"]
            )
        else:
            (x_train, y_train), val, (x_test, y_test) = loader.load_data(
                **config["load_kwargs"]
            )
        data = ((x_train, y_train), val, (x_test, y_test))
        return loader, data

    def load_fitable(
        self, loader: DataLoader, fitable_config: dict = None
    ) -> Union[Model, Tuner]:
        raise NotImplementedError

    def fit(self, fitable: Union[Model, Tuner], data: tuple) -> Union[Model, Tuner]:
        (x_train, y_train), val = data
        callbacks = [
            TensorBoard(**self.exp_config["tb_config"]),
            ReduceLROnPlateau(**self.exp_config["lr_config"]),
        ]
        if self.exp_config["builder_config"]["builder_type"] in [
            "siamese_builder",
            "classifier_builder",
        ]:
            callbacks.append(ClassificationMetrics(val))
        kwargs = dict(
            x=x_train,
            y=y_train,
            epochs=self.exp_config["run_config"]["epochs"],
            batch_size=self.exp_config["run_config"]["batch_size"],
            validation_data=val,
            class_weight=self.exp_config["run_config"]["class_weight"],
            callbacks=callbacks,
            verbose=self.exp_config["run_config"]["fit_verbosity"],
        )
        try:
            fitable.fit(**kwargs)
        except AttributeError as e:
            try:
                fitable.search(**kwargs)
            except AttributeError:
                raise ValueError(
                    "Param 'fitable' does not have a fit or a search method. Ensure"
                    "fitable is either of type 'Model' or 'Tuner'"
                )
        return fitable

    def test_fitable(self, fitable: Union[Model, Tuner], test_data: tuple) -> float:
        raise NotImplementedError

    def save_results(self, run: Run, fitable: Union[Model, Tuner]):
        raise NotImplementedError

    def new_model_path(self, i):
        model_path = Path(
            self.exp_config["run_config"]["model_path"]
        ).parent / "source_model_{}.h5".format(i)
        self.exp_config["run_config"]["model_path"] = model_path
        return model_path
