from sacred.run import Run
from tensorflow.keras.models import Model

from . import KerasJob
from ..callbacks import ClassificationMetrics


class Classification(KerasJob):
    def _fit(
        self, run: Run, fitable: Model, data: tuple, callbacks: list = None,
    ) -> Model:
        return super()._fit(
            run,
            fitable,
            data,
            callbacks=[ClassificationMetrics(data[-1], run.observers[0].dir + "/logs")],
        )
