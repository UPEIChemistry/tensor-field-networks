from copy import copy
from pathlib import Path

from sacred.run import Run
from tensorflow.keras.models import Model

from . import KerasJob
from .config_defaults import cm_config
from ..callbacks import CartesianMetrics


class Regression(KerasJob):
    pass


class StructurePrediction(Regression):
    @property
    def config_defaults(self):
        base = super().config_defaults
        base["loader_config"][
            "map_points"
        ] = False  # Ensure reconstruction works properly
        base["cm_config"] = copy(cm_config)
        return base

    def _fit(
        self, run: Run, fitable: Model, data: tuple, callbacks: list = None,
    ) -> Model:
        path = self.exp_config["run_config"]["root_dir"] / "cartesians"
        metric_data = data if self.exp_config["run_config"]["test"] else data[:2]
        return super()._fit(
            run,
            fitable,
            data,
            callbacks=[
                CartesianMetrics(path, *metric_data, **self.exp_config["cm_config"])
            ],
        )
