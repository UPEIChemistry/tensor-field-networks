import tensorflow as tf
from tensorflow.python.keras import backend as K, Sequential
from tensorflow.python.keras.layers import Layer, Dense
from tfn import utils
import numpy as np


class Convolution(Layer):

    def __init__(self,
                 image,
                 vectors,
                 radial=None,
                 si_units=16,
                 activation='relu',
                 num_filters=2,
                 **kwargs):

        super().__init__(dynamic=True, **kwargs)
        self.image = image
        self.vectors = vectors
        self.radial = radial
        self.si_units = si_units
        self.activation = activation
        self.num_filters = 2  # When ready, add num_filters here

        self._filters = []
        self._si_layer = None
        self._activation_layer = None

    @staticmethod
    def get_tensor_ro(tensor):

        return (tensor.shape[-1] - 1) / 2

    @staticmethod
    def cg_coefficient(size, axis, dtype=tf.float32):
        return tf.expand_dims(tf.eye(size, dtype=dtype), axis=axis)

    @staticmethod
    def lc_tensor(dtype=tf.float32):
        """
        Constant Levi-Civita tensor.
        """
        eijk_ = np.zeros((3, 3, 3))
        eijk_[0, 1, 2] = eijk_[1, 2, 0] = eijk_[2, 0, 1] = 1.
        eijk_[0, 2, 1] = eijk_[2, 1, 0] = eijk_[1, 0, 2] = -1.
        return tf.constant(eijk_, dtype=dtype)

    def call(self, inputs, **kwargs):
        conv_outputs = self.convolution(inputs)
        concat_outputs = self.concatenation(conv_outputs)
        si_outputs = self.self_interaction(concat_outputs)
        return self.equivariant_activation(si_outputs)

    @utils.inputs_to_dict
    def convolution(self, inputs):
        """Layer for computing rotationally-equivariant convolution on a set of input features.

                :param inputs: Flattened list of Tensors representing features.
                :return: List of output tensors
                """
        output_tensors = []
        for key, tensors in inputs.items():
            for i, tensor in enumerate(tensors):
                input_ro = self.get_tensor_ro(tensor)
                self._filters = [HarmonicFilter(radial=self.radial, filter_ro=n) for n in range(self.num_filters)]
                filter_outputs = [f([self.image, self.vectors]) for f in self._filters]

                # L x 0 -> L; shorthand for input_ro x filter_ro -> output_ro
                cg = self.cg_coefficient(tensor.shape[-1], axis=-2)
                output_tensors.append(tf.einsum('ijk,abfj,bfk->afi', cg, filter_outputs[0], tensor))
                if input_ro == 0:
                    # 0 x 1 -> 1
                    cg = self.cg_coefficient(3, axis=-1)
                    output_tensors.append(tf.einsum('ijk,abfj,bfk->afi', cg, filter_outputs[1], tensor))
                if input_ro == 1:
                    # 1 x 1 -> 0
                    cg = self.cg_coefficient(3, axis=0)
                    output_tensors.append(tf.einsum('ijk,abfj,bfk->afi', cg, filter_outputs[1], tensor))
                    # 1 x 1 -> 1
                    lc_tensor = self.lc_tensor()
                    output_tensors.append(tf.einsum('ijk,abfj,bfk->afi', lc_tensor, filter_outputs[1], tensor))

        return output_tensors

    @staticmethod
    @utils.inputs_to_dict
    def concatenation(inputs: list, axis=-2):
        return [K.concatenate(tensors, axis=axis) for tensors in inputs.values()]

    def self_interaction(self, inputs):
        self._si_layer = SelfInteraction(self.si_units)
        return self._si_layer(inputs)

    def equivariant_activation(self, inputs):
        self._activation_layer = EquivariantActivation(self.activation)
        return self._activation_layer(inputs)


class HarmonicFilter(Layer):
    """
    Abstract class for generating filters.

    :param radial: Callable. The learnable bits of an equivariant filter. Radial can be any tf callable
        (model, layer, op...) that takes the RBF image of shape (atoms, atoms, rbf) as input and combines it
        in some way with weights to return a learned tensor of shape (atoms, atoms, output_dim) that, when combined
        with a tensor derived from a spherical harmonic function aligned with provided unit vectors, returns a filter.
    :param num_filters: int. How many filters the generator should output. Supports up to 2 filters, currently.
    """
    def __init__(self,
                 radial=None,
                 filter_ro=0,
                 **kwargs):
        super().__init__(dynamic=True, **kwargs)
        if radial is None:
            radial = Sequential(
                [
                    Dense(16, dynamic=True),
                    Dense(16, dynamic=True)
                ],
            )
        self.radial = radial
        self.filter_ro = filter_ro

    @property
    def trainable_weights(self):
        if self.trainable:
            return super().trainable_weights + self.radial.trainable_weights
        else:
            return []

    def call(self, inputs, **kwargs):
        """Generate the filter based on provided image (and vectors, depending on requested filter rotation order).

        :param inputs: List of input tensors including image, of shape (atoms, atoms, rbf), and unit_vectors, of shape
            (atoms, atoms, 3).
        :return: tensor. HarmonicFilter of specified rotation order.
        """
        image, vectors = inputs
        if self.filter_ro == 0:
            # [N, N, output_dim, 1]
            return K.expand_dims(self.radial(image), axis=-1)
        elif self.filter_ro == 1:
            masked_radial = self.mask_radial(self.radial(image), vectors)
            # [N, N, output_dim, 3]
            return K.expand_dims(vectors, axis=-2) * K.expand_dims(masked_radial, axis=-1)
        elif self.filter_ro == 2:
            masked_radial = self.mask_radial(self.radial(image), vectors)
            # [N, N, output_dim, 5]
            return K.expand_dims(self.l2_spherical_harmonic(vectors), axis=-2) * K.expand_dims(masked_radial, axis=-1)

    @staticmethod
    def mask_radial(radial, vectors):
        norm = tf.norm(vectors, axis=-1)
        condition = K.expand_dims(norm < K.epsilon(), axis=-1)
        tile = K.tile(condition, [1, 1, radial.shape[-1]])

        # [N, N, output_dim]
        return tf.where(tile, K.zeros_like(radial), radial)

    @staticmethod
    def l2_spherical_harmonic(tensor):
        """

        :param tensor: must be of shape [atoms, atoms, 3]
        :return: tensor. Result of L2 spherical harmonic function applied to input tensor
        """
        x = tensor[:, :, 0]
        y = tensor[:, :, 1]
        z = tensor[:, :, 2]
        r2 = tf.maximum(tf.reduce_sum(tf.square(tensor), axis=-1), K.epsilon())
        # return : [N, N, 5]
        output = tf.stack([x * y / r2,
                           y * z / r2,
                           (-tf.square(x) - tf.square(y) + 2. * tf.square(z)) / (2 * np.sqrt(3) * r2),
                           z * x / r2,
                           (tf.square(x) - tf.square(y)) / (2. * r2)],
                          axis=-1)
        return output


class EquivarantWeighted(Layer):

    def __init__(self,
                 **kwargs):
        super().__init__(dynamic=True, **kwargs)
        self.weight_dict = {}

    def add_weight_to_nested_dict(self, indices, *args, **kwargs):
        def add_to_dict(current_dict, _indices):
            if len(_indices) > 1:
                new_dict = current_dict.setdefault(_indices.pop(0), {})
                add_to_dict(new_dict, _indices)
            else:
                current_dict[_indices[0]] = self.add_weight(*args, **kwargs)

        indices = list(indices)
        add_to_dict(self.weight_dict, indices)


class SelfInteraction(EquivarantWeighted):

    def __init__(self,
                 units: int,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units

    @utils.shapes_to_dict
    def build(self, input_shape):
        for (key, shapes) in input_shape.items():
            for i, shape in enumerate(shapes):
                self.add_weight_to_nested_dict(
                    [key, i],
                    name='SIKernel_RO{}_I{}'.format(str(key), str(i)),
                    shape=(self.units, shape[-2]),
                )
        self.built = True

    @utils.inputs_to_dict
    def call(self, inputs, **kwargs):
        output_tensors = []
        for key, tensors in inputs.items():
            for i, tensor in enumerate(tensors):
                w = self.weight_dict[key][i]
                output_tensors.append(
                    tf.transpose(tf.einsum('afi,gf->aig', tensor, w), perm=[0, 2, 1])
                )
        return output_tensors


class EquivariantActivation(EquivarantWeighted):

    def __init__(self,
                 activation=None,
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(activation, str):
            activation = tf.keras.activations.get(activation)
        elif activation is None:
            activation = utils.shifted_softplus
        self.activation = activation

    @utils.shapes_to_dict
    def build(self, input_shape):
        for (key, shapes) in input_shape.items():
            for i, shape in enumerate(shapes):
                self.add_weight_to_nested_dict(
                    [key, i],
                    name='RTSBias_RO{}_I{}'.format(str(key), str(i)),
                    shape=(shape[-2],),
                )
        self.built = True

    @utils.inputs_to_dict
    def call(self, inputs, **kwargs):
        output_tensors = []
        for key, tensors in inputs.items():
            for i, tensor in enumerate(tensors):
                b = self.weight_dict[key][i]
                if key == 0:
                    b = tf.expand_dims(tf.expand_dims(b, axis=0), axis=-1)
                    output_tensors.append(
                        self.activation(tensor + b)
                    )
                elif key == 1:
                    norm = utils.norm_with_epsilon(tensor, axis=-1)
                    a = self.activation(
                        K.bias_add(norm, b)
                    )
                    output_tensors.append(
                        tensor * tf.expand_dims(a / norm, axis=-1)
                    )
        return output_tensors
