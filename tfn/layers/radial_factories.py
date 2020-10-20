import json

import tensorflow as tf
from tensorflow.keras import activations, regularizers, Sequential
from tensorflow.keras.layers import Layer


class RadialFactory(object):
    """
    Abstract class for RadialFactory objects, defines the interface. Subclass
    """

    def __init__(
        self,
        num_layers: int = 2,
        units: int = 32,
        activation: str = "ssp",
        l2_lambda: float = 0.0,
        **kwargs,
    ):
        self.num_layers = num_layers
        self.units = units
        if activation is None:
            activation = "ssp"
        if isinstance(activation, str):
            self.activation = activation
        else:
            raise ValueError(
                "Expected `str` for param `activation`, but got `{}` instead. "
                "Ensure `activation` is a string mapping to a valid keras "
                "activation function"
            )
        self.l2_lambda = l2_lambda
        self.sum_points = kwargs.pop("sum_points", False)
        self.dispensed_radials = 0

    def get_radial(self, feature_dim, input_order=None, filter_order=None):
        raise NotImplementedError

    def to_json(self):
        self.__dict__["type"] = type(self).__name__
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
        layers = [
            Radial(
                self.units,
                self.activation,
                self.l2_lambda,
                sum_points=self.sum_points,
                name=f"radial_{self.dispensed_radials}/layer_{i}",
            )
            for i in range(self.num_layers)
        ]
        layers.append(
            Radial(
                feature_dim,
                self.activation,
                self.l2_lambda,
                sum_points=self.sum_points,
                name=f"radial_{self.dispensed_radials}/layer_{self.num_layers}",
            )
        )

        self.dispensed_radials += 1
        return Sequential(layers)

    @classmethod
    def from_json(cls, config: str):
        return cls(**json.loads(config))


class Radial(Layer):
    def __init__(
        self, units: int = 32, activation: str = "ssp", l2_lambda: float = 0.0, **kwargs
    ):
        self.sum_points = kwargs.pop("sum_points", False)
        super().__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.l2_lambda = l2_lambda
        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[-1], self.units),
            regularizer=regularizers.l2(self.l2_lambda),
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            regularizer=regularizers.l2(self.l2_lambda),
        )
        self.built = True

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(list(input_shape)[:-1] + [self.units])

    def get_config(self):
        base = super().get_config()
        updates = dict(units=self.units, activation=self.activation,)
        return {**base, **updates}

    def call(self, inputs, training=None, mask=None):
        equation = "bpf,fu->bpu" if self.sum_points else "bpqf,fu->bpqu"
        return self.activation(tf.einsum(equation, inputs, self.kernel) + self.bias)
