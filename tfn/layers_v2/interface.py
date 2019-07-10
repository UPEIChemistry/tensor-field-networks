import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from .core import HarmonicFilter
from .. import utils
import numpy as np


class EquivariantLayer(Layer):

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
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

    @staticmethod
    def get_tensor_ro(tensor):

        return (tensor.shape[-1] - 1) / 2

    @staticmethod
    def cg_coefficient(size, axis, dtype=tf.float32):
        return tf.expand_dims(tf.eye(size, dtype=dtype), axis=axis)

    @staticmethod
    def lc_tensor(dtype=tf.float32):
        """
        Constant Levi-Civita tensor

        Returns:
            K.Tensor of shape [3, 3, 3]
        """
        eijk_ = np.zeros((3, 3, 3))
        eijk_[0, 1, 2] = eijk_[1, 2, 0] = eijk_[2, 0, 1] = 1.
        eijk_[0, 2, 1] = eijk_[2, 1, 0] = eijk_[1, 0, 2] = -1.
        return tf.constant(eijk_, dtype=dtype)


class Convolution(EquivariantLayer):

    def __init__(self,
                 image,
                 vectors,
                 radial=None,
                 num_filters=2,
                 **kwargs):

        super(Convolution, self).__init__(**kwargs)
        self.image = image
        self.vectors = vectors
        self.radial = radial
        self.num_filters = 2  # When ready, add num_filters here

    @utils.inputs_to_dict
    def call(self, inputs, **kwargs):
        """Layer for computing rotationally-equivariant convolution on a set of input features.

        :param inputs: Flattened list of Tensors representing features.
        :return: List of output tensors
        """
        output_tensors = []
        for key, tensors in inputs.items():
            for i, tensor in enumerate(tensors):
                input_ro = self.get_tensor_ro(tensor)
                filters = []
                for n in range(self.num_filters):
                    f = HarmonicFilter(radial=self.radial, filter_ro=n)
                    filters.append(f([self.image, self.vectors]))
                    # self.trainable_weights.extend(f.trainable_weights)
                # L x 0 -> L; shorthand for input_ro x filter_ro -> output_ro
                cg = self.cg_coefficient(tensor.shape[-1], axis=-2)
                output_tensors.append(tf.einsum('ijk,abfj,bfk->afi', cg, filters[0], tensor))
                if input_ro == 0:
                    # 0 x 1 -> 1
                    cg = self.cg_coefficient(3, axis=-1)
                    output_tensors.append(tf.einsum('ijk,abfj,bfk->afi', cg, filters[1], tensor))
                if input_ro == 1:
                    # 1 x 1 -> 0
                    cg = self.cg_coefficient(3, axis=0)
                    output_tensors.append(tf.einsum('ijk,abfj,bfk->afi', cg, filters[1], tensor))
                    # 1 x 1 -> 1
                    lc_tensor = self.lc_tensor()
                    output_tensors.append(tf.einsum('ijk,abfj,bfk->afi', lc_tensor, filters[1], tensor))

        return output_tensors
