import socket
from abc import ABCMeta, abstractmethod
from copy import copy
from pathlib import Path
from typing import List

from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver, RunObserver
from tensorflow.keras.models import Model

from tfn.tools.ingredients import builder_ingredient, data_ingredient
from tfn.tools.jobs import config_defaults as cd


class Job(metaclass=ABCMeta):
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

    @property
    def experiment(self):
        """
        Experiment object required for Sacred.

        :return: sacred.Experiment object.
        """
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

    @abstractmethod
    def _main(self, run, seed, fitable, fitable_config, loader_config):
        """
        Private method containing the actual work completed by the job. Implemented is a default
        workflow for a basic keras/kerastuner type job.

        :param run: sacred.Run object. See sacred documentation for more details on utility.
        :param fitable: Optional tensorflow.keras.Model or kerastuner.Tuner object.
            Model-like which contains a fit method.
        :param fitable_config: Optional dict. Contains data which can be used to create a new
            fitable instance.
        :param loader_config: Optional dict. Contains data which can be used to create a new
            DataLoader instance.
        """
        pass

    def run(
        self,
        fitable: Model = None,
        fitable_config: dict = None,
        loader_config: dict = None,
    ):
        """
        Exposed method of the particular job. Runs whatever work is entailed by the job based on
        the content provided in `self.exp_config`.
        """

        @self.experiment.main
        def main(_run, _seed):
            self.exp_config["run_config"]["root_dir"] = Path(
                _run.observers[0].dir
            ).absolute()
            self._main(_run, _seed, fitable, fitable_config, loader_config)

        self.experiment.run()

    @abstractmethod
    def _load_data(self, config):
        """
        Obtains a loader using ingredients.get_loader and self.exp_config['loader_config']

        :return: Loader object and the data returned by that Loader's get_data method.
        """
        pass

    @abstractmethod
    def _load_fitable(self, loader, fitable_config):
        """
        Defines and compiles a fitable (keras.model or keras_tuner.tuner) which implements
        a 'fit' method. This method calls either get_builder, or get_hyper_factory, depending on
        which type of fitable is beind loaded.

        :return: Model or Tuner object.
        """
        pass

    @abstractmethod
    def _fit(self, run, fitable, data, callbacks):
        """

        :param run: sacred.Run object. See sacred documentation for details on utility.
        :param fitable: tensorflow.keras.Model object.
        :param data: tuple. train and validation data in the form (train, val), where train is
            the tuple (x_train, y_train).
        :param callbacks: Optional list. List of tensorflow.keras.Callback objects to pass to
            fitable.fit method.
        :return: tensorflow.keras.Model object.
        """
        pass

    @abstractmethod
    def _test_fitable(self, run, fitable, test_data):
        """
        :param fitable: tensorflow.keras.Model object.
        :param test_data: tuple. contains (x_test, y_test).
        :return: float. Scalar test_loss value.
        """
        pass

    @abstractmethod
    def _save_fitable(self, run, fitable):
        """
        :param run: sacred.Run object. see sacred documentation for more details on utility.
        :param fitable: tensorflow.keras.Model object.
        """
        pass

    @abstractmethod
    def _new_model_path(self, i):
        pass

    @property
    def config_defaults(self):
        """
        Defines default values for the various config dictionaries required for the Job.

        :return: dict. Experiment dictionary containing necessary config(s) for the Job.
        """
        return {
            "ingredients": [data_ingredient, builder_ingredient],
            "run_config": copy(cd.run_config),
            "loader_config": copy(cd.loader_config),
            "builder_config": copy(cd.builder_config),
            "tb_config": copy(cd.tb_config),
            "lr_config": copy(cd.lr_config),
        }
