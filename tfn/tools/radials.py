import json

import tensorflow as tf
from tensorflow.python.keras import Sequential, activations, regularizers
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.models import model_from_json
from tfn.layers import DenseRadialFactory, RadialFactory


def get_radial_factory(identifier='multi_dense', radial_kwargs: dict = None):
    radial = None
    if identifier == 'multi_dense':
        radial = DenseRadialFactory
    elif identifier == 'single_dense':
        radial = SingleModelDenseRadialFactory
    elif identifier == 'single_conv':
        radial = SingleModelConvRadialFactory
    elif identifier == 'multi_conv':
        radial = ConvRadialFactory

    if radial_kwargs is not None:
        return radial(**radial_kwargs)
    else:
        return radial()


class SingleModelRadialFactory(RadialFactory):
    def __init__(self,
                 *args,
                 **kwargs):
        self.radial = kwargs.pop('radial', None)
        super().__init__(*args, **kwargs)

    def to_json(self):
        self.__dict__['radial'] = None
        return super().to_json()

    @classmethod
    def from_json(cls, config: str):
        config = json.loads(config)
        if config['radial']:
            config['radial'] = model_from_json(config['radial'])
        else:
            config['radial'] = None
        return cls(**config)


class SingleModelDenseRadialFactory(SingleModelRadialFactory, DenseRadialFactory):
    def get_radial(self, feature_dim, input_order=None, filter_order=None):
        if self.radial is None:
            self.radial = super().get_radial(feature_dim, input_order, filter_order)
        return self.radial


class ConvRadialFactory(DenseRadialFactory):
    """
    input: (mols, atoms, atoms, 80), output: (mols, atoms, atoms, feature_dim)
    """

    def __init__(self, *args, **kwargs):
        self.kernel_size = kwargs.pop('kernel_size', 3)
        super().__init__(*args, **kwargs)

    def get_radial(self, feature_dim, input_order=None, filter_order=None):
        return Sequential([
                              Conv2D(
                self.units,
                kernel_size=self.kernel_size,
                padding='same',
                activation=activations.get(self.activation),
                kernel_regularizer=regularizers.l2(self.kernel_lambda),
                bias_regularizer=regularizers.l2(self.bias_lambda),
            )
            for _ in range(self.num_layers)
        ] + [
            Conv2D(
                feature_dim,
                kernel_size=self.kernel_size,
                padding='same',
                activation=activations.get(self.activation),
                kernel_regularizer=regularizers.l2(self.kernel_lambda),
                bias_regularizer=regularizers.l2(self.bias_lambda),
            )
        ])


class SingleModelConvRadialFactory(SingleModelRadialFactory, ConvRadialFactory):
    def get_radial(self, feature_dim, input_order=None, filter_order=None):
        if self.radial is None:
            self.radial = super().get_radial(feature_dim, input_order, filter_order)
        return self.radial


tf.keras.utils.get_custom_objects().update({
    SingleModelRadialFactory.__name__: SingleModelRadialFactory,
    SingleModelDenseRadialFactory.__name__: SingleModelDenseRadialFactory,
    ConvRadialFactory.__name__: ConvRadialFactory,
    SingleModelConvRadialFactory.__name__: SingleModelConvRadialFactory
})
