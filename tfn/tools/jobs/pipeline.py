from pathlib import Path
from typing import List

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import InputLayer
from tfn.layers.atomic_images import Unstandardization

from . import KerasJob


class Pipeline(KerasJob):
    NONTRANSFERABLE_LAYERS = (Unstandardization, InputLayer)
    BLACKLISTED_LAYER_NAMES = ["embedding"]

    def __init__(self, jobs: List[KerasJob], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jobs = jobs

    def _main(self, run, seed, fitable=None, loader_config=None, fitable_config=None):
        model_path = None
        for i, job in enumerate(self.jobs):
            loader, _ = job._load_data()
            fitable = job._load_fitable(loader)
            try:
                fitable = self.initialize_fitable_weights(fitable, model_path)
            except (FileNotFoundError, TypeError) as _:
                pass

            model_path = self._new_model_path(f"model_{i}.h5")
            job.exp_config["run_config"]["model_path"] = model_path
            job.exp_config["run_config"]["root_dir"] = (
                self.exp_config["run_config"]["root_dir"] / f"pipeline_model_{i}"
            )
            fitable = job._main(run, seed, fitable)

        return fitable

    def layer_is_valid(self, layer):
        if layer is None:
            return False
        elif any(
            [
                isinstance(layer, self.NONTRANSFERABLE_LAYERS),
                any([layer.name in name for name in self.BLACKLISTED_LAYER_NAMES]),
            ]
        ):
            return False
        else:
            return True

    def initialize_fitable_weights(self, fitable: Model, path) -> Model:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"hdf5 file {path} does not exist - cannot read weights."
            )
        temp_path = self.exp_config["run_config"]["root_dir"] / "temp_model.h5"
        fitable.save(temp_path)
        fitable = load_model(temp_path)
        fitable.load_weights(path, by_name=True, skip_mismatch=True)
        return fitable
