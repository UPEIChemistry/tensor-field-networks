import json

import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tfn.layers import DenseRadialFactory


def get_radial_factory(identifier="multi_dense", radial_kwargs: dict = None):
    radial = None
    if identifier == "single_dense":
        radial = SingleModelDenseRadialFactory
    else:
        radial = DenseRadialFactory

    if radial_kwargs is not None:
        return radial(**radial_kwargs)
    else:
        return radial()


class SingleModelDenseRadialFactory(DenseRadialFactory):
    def __init__(self, *args, **kwargs):
        self.radial = kwargs.pop("radial", None)
        super().__init__(*args, **kwargs)

    def to_json(self):
        self.__dict__["radial"] = None
        return super().to_json()

    @classmethod
    def from_json(cls, config: str):
        config = json.loads(config)
        if config["radial"]:
            config["radial"] = model_from_json(config["radial"])
        else:
            config["radial"] = None
        return cls(**config)

    def get_radial(self, feature_dim, input_order=None, filter_order=None):
        if self.radial is None:
            self.radial = super().get_radial(feature_dim, input_order, filter_order)
        return self.radial


tf.keras.utils.get_custom_objects().update(
    {SingleModelDenseRadialFactory.__name__: SingleModelDenseRadialFactory,}
)
