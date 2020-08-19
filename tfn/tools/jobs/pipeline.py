from copy import copy

from h5py import File
from tensorflow.keras.models import Model
from atomic_images.layers import Unstandardization

from . import KerasJob
from .config_defaults import pipeline_config, builder_config


class Pipeline(KerasJob):
    NONTRANSFERABLE_LAYERS = (Unstandardization,)
    BLACKLISTED_LAYER_NAMES = ["embedding"]

    @property
    def config_defaults(self):
        config = copy(super().config_defaults)
        config["pipeline_config"] = copy(pipeline_config)
        config["builder_config"] = copy(builder_config)
        return config

    def _main(self, run, fitable=None, loader_config=None, fitable_config=None):
        model_path = None
        for i, config in enumerate(self.exp_config["pipeline_config"]["configs"]):
            config = self.add_config_defaults(config)
            loader_config = config["loader_config"]
            fitable_config = config["builder_config"]
            if i == 0:
                model_path = self._new_model_path(i)
                super()._main(
                    run, loader_config=loader_config, fitable_config=fitable_config,
                )
            else:
                loader, _ = self._load_data(loader_config)
                fitable = self._load_fitable(loader, fitable_config)
                fitable = self.initialize_fitable_weights(fitable, model_path)
                model_path = self._new_model_path(i)
                super()._main(
                    run,
                    fitable=fitable,
                    loader_config=loader_config,
                    fitable_config=fitable_config,
                )

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
        layer_dict = {layer.name: layer for layer in fitable.layers}
        with File(path, "r") as f:
            for old_layer_name in list(f["model_weights"].keys()):
                layer = layer_dict.get(old_layer_name, None)
                if not self.layer_is_valid(layer):
                    continue
                weight_names = f["model_weights"][old_layer_name].attrs["weight_names"]
                weights = [f["model_weights"][old_layer_name][i] for i in weight_names]
                layer.set_weights(weights)
                layer.trainable = not self.exp_config["pipeline_config"][
                    "freeze_layers"
                ]
        return fitable
