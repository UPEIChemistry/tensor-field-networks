from sacred.run import Run
from tensorflow.keras.models import load_model, Model

from . import KerasJob
from ..loaders import DataLoader


class LoadModel(KerasJob):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        path = self.exp_config["run_config"]["model_path"]
        print(f"Loading pre-trained model from file {path}")
        self.model = load_model(path)

    def _main(
        self,
        run: Run,
        seed: int,
        fitable: Model = None,
        fitable_config: dict = None,
        loader_config: dict = None,
    ):
        self._save_fitable(run, self.model)
        return self.model

    def _load_fitable(self, loader: DataLoader, fitable_config: dict = None) -> Model:
        return self.model
