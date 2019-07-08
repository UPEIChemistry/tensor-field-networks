import tensorflow as tf
from keras.layers import Layer, Lambda
from keras import backend as K
import numpy as np

from .. import utils


class Filter(Layer):
    """Abstract class for generating filters.

    """

    def __init__(self,
                 activation,
                 kernel_dict,
                 bias_dict,
                 **kwargs):
        super().__init__(**kwargs)
        self.activation = activation
        self.kernel_dict = kernel_dict
        self.bias_dict = bias_dict

        self.image = None
        self.unit_vectors = None

    def compute_output_shape(self, input_shape):
        # [N, N, rbf], [N, 3]
        rbf_shape, unit_vectors_shape = input_shape
        n, _, rbf = rbf_shape
        f0_output = int(self.kernel_dict[0][1].shape[0])
        f1_output = int(self.kernel_dict[1][1].shape[0])
        return [
            (n, n, f0_output, 1),
            (n, n, f1_output, 3)
        ]

    def call(self, inputs, **kwargs):
        self.image, self.unit_vectors = inputs

        # TODO: If adding more filter support, add here!
        return [
            self.get_filter0(self.kernel_dict[0].values(), self.bias_dict[0].values()),
            self.get_filter1(self.kernel_dict[1].values(), self.bias_dict[1].values())
        ]

    def get_filter0(self, weights, biases):
        radial = self.get_radial(weights, biases)
        # [N, N, output_dim, 1]
        return K.expand_dims(radial, axis=-1)

    def get_filter1(self, weights, biases):
        masked_radial = self.get_masked_radial(weights, biases)
        # [N, N, otuput_dim, 3]
        return (K.expand_dims(self.unit_vectors, axis=-2)
                * K.expand_dims(masked_radial, axis=-1))

    def get_filter2(self, weights, biases):
        masked_radial = self.get_masked_radial(weights, biases)
        # [N, N, output_dim, 5]
        return (K.expand_dims(L2SphericalHarmonic()(self.unit_vectors), axis=-2)
                * K.expand_dims(masked_radial, axis=-1))

    # TODO: This is currently the filter generating NN!
    def get_radial(self,
                   weights,
                   biases):
        w1, w2 = weights
        b1, b2 = biases
        hidden_layer = TensorDot(axes=[[2], [1]], name='RadialHiddenTensorDot')([self.image, w1])
        hidden_layer = self.activation(b1 + hidden_layer)
        radial = TensorDot(axes=[[2], [1]], name='RadialTensorDot')([hidden_layer, w2])
        radial = b2 + radial

        # [N, N, output_dim]
        return radial

    def get_masked_radial(self, weights, biases):
        radial = self.get_radial(weights, biases)
        if self.unit_vectors is None:
            raise ValueError('Filters of rotation 1 one must be passed a distance matrix.'
                             ' Ensure dist_matrix arg is passed to constructor')
        else:
            norm = Normalize(keepdims=False)(self.unit_vectors)
            condition = K.expand_dims(norm < K.epsilon(), axis=-1)
            tile = K.tile(condition, [1, 1, radial.shape[-1]])

            # [N, N, output_dim]
            return K.switch(tile, K.zeros_like(radial), radial)


# FIXME: This class is a bit of a mess current, could use some refactoring
class EquivariantCombination(Layer):

    def call(self, inputs, **kwargs):
        # Filter shapes and ROs: [..., 2l + 1] = l
        tensor, f = inputs
        fro = utils.get_l_shape(int(f.shape[-1]))
        iro = utils.get_l_shape(int(tensor.shape[-1]))

        if fro == 0:
            return self._ilf0ol(tensor, f)
        elif iro == 0 and fro == 1:
            return self._i0f1o1(tensor, f)
        elif iro == 1 and fro == 1:
            return [
                self._i1f1o0(tensor, f),
                self._i1f1o1(tensor, f)
            ]

    # i,j,k = x,y,z indices, a, b = atoms, f = filters
    def _ilf0ol(self, tensor, f):
        cg = self.cg_coefficient(int(tensor.shape[-1]), axis=-2)
        return [EinSum('ijk,abfj,bfk->afi')([cg, f, tensor])]

    def _i0f1o1(self, tensor, f):
        cg = self.cg_coefficient(3, axis=-1)
        return [EinSum('ijk,abfj,bfk->afi')([cg, f, tensor])]

    def _i1f1o0(self, tensor, f):
        cg = self.cg_coefficient(1, axis=0)
        return EinSum('ijk,abfj,bfk->afi')([cg, f, tensor])

    def _i1f1o1(self, tensor, f):
        lc_tensor = self.lc_tensor()
        return EinSum('ijk,abfj,bfk->afi')([lc_tensor, f, tensor])

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
        eijk_ = np.zeros((3, 3, 3))
        eijk_[0, 1, 2] = eijk_[1, 2, 0] = eijk_[2, 0, 1] = 1.
        eijk_[0, 2, 1] = eijk_[2, 1, 0] = eijk_[1, 0, 2] = -1.
        return K.constant(eijk_, dtype=dtype)


class DifferenceMatrix(Layer):

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape.insert(-2, input_shape[-2])
        return tuple(input_shape)

    # [..., N, 3] as input shape where N is num_atoms, returns [..., N, N, 3]
    def call(self, inputs, **kwargs):
        i = K.expand_dims(inputs, axis=-2)
        j = K.expand_dims(inputs, axis=-3)
        return i - j


class UnitVectors(Layer):

    def call(self, inputs, **kwargs):
        v = inputs
        den = Normalize()(v)
        return v / den


class Normalize(Layer):

    def __init__(self,
                 axis=-1,
                 keepdims=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, inputs, **kwargs):

        return K.sqrt(
            K.maximum(K.sum(K.square(inputs), axis=self.axis, keepdims=self.keepdims), K.epsilon())
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
        def einsum(inputs):
            return tf.einsum(equation, *inputs)
        super().__init__(einsum, **kwargs)


class TensorDot(Lambda):

    def __init__(self, axes, **kwargs):
        def tensordot(inputs):
            return tf.tensordot(*inputs, axes=axes)
        super().__init__(tensordot, **kwargs)
