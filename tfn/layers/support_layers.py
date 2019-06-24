import tensorflow as tf
from keras.layers import Layer, Lambda
from keras import backend as K
from functools import partial

from .. import utils


class Filter(Layer):
    """Abstract class for generating filters.

    """

    def __init__(self,
                 dist_matrix,  # TODO: It's possible that this needs to be a diff matrix, not a dist matrix
                 hidden_dim,
                 output_dim,
                 activation,
                 layer_weights=None,
                 layer_biases=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.dist_matrix = dist_matrix
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        if layer_weights is None:
            raise ValueError('Filter layer must be passed list of weight tensors divisible by two (two for each '
                             'output filter)')
        self.layer_weights = layer_weights
        if layer_biases is None:
            layer_biases = [0 for _ in range(len(layer_weights))]
        self.layer_biases = layer_biases

    def call(self, inputs, **kwargs):
        return [
            self.get_filter0(inputs,
                             self.layer_weights[0],
                             self.layer_weights[1],
                             self.layer_biases[0],
                             self.layer_biases[1]
                             ),
            self.get_filter1(inputs,
                             self.layer_weights[2],
                             self.layer_weights[3],
                             self.layer_biases[2],
                             self.layer_biases[3]
                             )  # TODO: If adding more filter support, add here!
        ]

    def get_radial(self, tensor, w1, w2, b1=0, b2=0):
        hidden_layer = Lambda(tf.tensordot, name='RadialHiddenTensorDot')([tensor, w1, [[2], [1]]])
        hidden_layer = self.activation(b1 + hidden_layer)
        radial = Lambda(tf.tensordot, name='RadialTensorDot')([hidden_layer, w2, [[2], [1]]])
        radial = b2 + radial

        # [N, N, output_dim]
        return radial

    def get_masked_radial(self, *args):
        radial = self.get_radial(*args)
        if self.dist_matrix is None:
            raise ValueError('Filters of rotation 1 one must be passed a distance matrix.'
                             ' Ensure dist_matrix arg is passed to constructor')
        else:
            norm = K.l2_normalize(self.dist_matrix, axis=-1)
            condition = K.tile(K.expand_dims(norm < K.epsilon(), axis=-1), [1, 1, self.output_dim])

            return K.switch(condition, K.zeros_like(radial), radial)

    def get_filter0(self, *args):
        radial = self.get_radial(*args)
        return K.expand_dims(radial, axis=-1)

    def get_filter1(self, *args):
        masked_radial = self.get_masked_radial(*args)
        return (K.expand_dims(UnitVectors()(self.dist_matrix), axis=-2)
                * K.expand_dims(masked_radial, axis=-1))

    def get_filter2(self, *args):
        masked_radial = self.get_masked_radial(*args)
        return (K.expand_dims(L2SphericalHarmonic()(self.dist_matrix), axis=-2)
                * K.expand_dims(masked_radial, axis=-1))


class EquivariantCombination(Layer):

    def call(self, inputs, **kwargs):
        # Filter shapes and ROs: [..., 2l + 1] = l
        tensor, f = inputs
        fro = utils.get_l_shape(f.shape[-1])
        iro = utils.get_l_shape(tensor.shape[-1])

        if fro == 0:
            return self._ilf0ol(tensor, f)
        elif iro == 0 and fro == 1:
            return self._i0f1o1(tensor, f)
        elif iro == 1 and fro == 1:
            return

    def _ilf0ol(self, tensor, f):
        cg = self.cg_coefficient(tensor.shape[-1], axis=-2)
        return [EinSum('ijk,abfj,bfk->afi')(cg, f, tensor)]

    def _i0f1o1(self, tensor, f):
        cg = self.cg_coefficient(tensor.shape[-1], axis=-1)
        return [EinSum('ijk,abfj,bfk->afi')(cg, f, tensor)]

    def _i1f1o1(self, tensor, f):
        cg = self.cg_coefficient(tensor.shape[-1], axis=0)
        lc_tensor = self.lc_tensor()

        return [
            EinSum('ijk,abfj,bfk->afi')(cg, f, tensor),
            EinSum('ijk,abfj,bfk->afi')(lc_tensor, f, tensor)
        ]

    @staticmethod
    def cg_coefficient(size, axis):
        return K.expand_dims(K.eye(size=size), axis=axis)

    @staticmethod
    def lc_tensor(dtype='float32'):
        """
        Constant Levi-Civita tensor

        Returns:
            K.Tensor of shape [3, 3, 3]
        """
        eijk_ = K.zeros((3, 3, 3))
        eijk_[0, 1, 2] = eijk_[1, 2, 0] = eijk_[2, 0, 1] = 1.
        eijk_[0, 2, 1] = eijk_[2, 1, 0] = eijk_[1, 0, 2] = -1.
        return K.constant(eijk_, dtype=dtype)


class UnitVectors(Layer):

    def call(self, inputs, **kwargs):
        v = inputs
        den = Normalize()(v)
        return v / den


class Normalize(Layer):

    def __init__(self,
                 axis=None,
                 keep_dims=True,
                 **kwargs):
        super().__init__(**kwargs)
        if axis is None:
            axis = -1
        self.axis = axis
        self.keep_dims = keep_dims

    def call(self, inputs, **kwargs):

        return K.sqrt(
            K.maximum(K.sum(K.square(inputs), axis=self.axis, keepdims=self.keep_dims), K.epsilon())
        )


class L2SphericalHarmonic(Layer):

    def call(self, inputs, **kwargs):
        rij = inputs
        x, y, z = rij[:, :, :3]
        r2 = K.maximum(K.sum(K.square(rij), axis=-1), K.epsilon())
        return K.stack([x * y / r2,
                        y * z / r2,
                        (-K.square(x) - K.square(y) + 2. * K.square(z)) / (2 * K.sqrt(3) * r2),
                        z * x / r2,
                        (K.square(x) - K.square(y)) / (2. * r2)],
                       axis=-1)


class EinSum(Lambda):

    def __init__(self, equation, **kwargs):
        super().__init__(None, **kwargs)
        self.function = partial(tf.einsum, equation)

    def call(self, *inputs, mask=None):  # TODO: Overriding this method may cause problems...
        arguments = self.arguments
        return self.function(*inputs, **arguments)
