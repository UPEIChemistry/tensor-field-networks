import json

from tensorflow.keras import activations, regularizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


class RadialFactory(object):
    """
    Abstract class for RadialFactory objects, defines the interface. Subclass
    """
    def __init__(self,
                 num_layers: int = 2,
                 units: int = 32,
                 activation: str = 'ssp',
                 kernel_lambda: float = 0.,
                 bias_lambda: float = 0.,
                 **kwargs):
        self.num_layers = num_layers
        self.units = units
        if activation is None:
            activation = 'ssp'
        if isinstance(activation, str):
            self.activation = activation
        else:
            raise ValueError('Expected `str` for param `activation`, but got `{}` instead. '
                             'Ensure `activation` is a string mapping to a valid keras '
                             'activation function')
        self.kernel_lambda = kernel_lambda
        self.bias_lambda = bias_lambda

    def get_radial(self, feature_dim, input_order=None, filter_order=None):
        raise NotImplementedError

    def to_json(self):
        self.__dict__['type'] = type(self).__name__
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, config: str):
        raise NotImplementedError


class DenseRadialFactory(RadialFactory):
    """
    Default factory class for supplying radial functions to a Convolution layer. Subclass this
    factory and override its `get_radial` method to return custom radial instances/templates.
    You must also override the `to_json` and `from_json` and register any custom `RadialFactory`
    classes to a unique string in the keras global custom objects dict.
    """
    def get_radial(self, feature_dim, input_order=None, filter_order=None):
        """
        Factory method for obtaining radial functions of a specified architecture, or an instance
        of a radial function (i.e. object which inherits from Layer).

        :param feature_dim: Dimension of the feature tensor being point convolved with the filter
            produced by this radial function. Use to ensure radial function outputs a filter of
            shape (points, feature_dim, filter_order)
        :param input_order: Optional. Rotation order of the of the feature tensor point convolved
            with the filter produced by this radial function
        :param filter_order: Optional. Rotation order of the filter being produced by this radial
            function.
        :return: Keras Layer object, or subclass of Layer. Must have attr dynamic == True and
            trainable == True.
        """
        return Sequential([
              Dense(
                  self.units,
                  activation=activations.get(self.activation),
                  kernel_regularizer=regularizers.l2(self.kernel_lambda),
                  bias_regularizer=regularizers.l2(self.bias_lambda),
              )
              for _ in range(self.num_layers)
          ] + [
              Dense(
                  feature_dim,
                  activation=activations.get(self.activation),
                  kernel_regularizer=regularizers.l2(self.kernel_lambda),
                  bias_regularizer=regularizers.l2(self.bias_lambda),
              )
          ])

    @classmethod
    def from_json(cls, config: str):
        return cls(**json.loads(config))
