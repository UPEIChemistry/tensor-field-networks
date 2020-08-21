from tensorflow.keras.layers import Layer

from .atomic_images import DummyAtomMasking

from . import Convolution, SelfInteraction, EquivariantActivation


class MolecularLayer(Layer):
    def build(self, input_shape):
        if len(input_shape) < self.total_required_inputs:
            raise ValueError(
                "Ensure one_hot tensor is passed before other relevant tensors"
            )
        _, *input_shape = input_shape
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """
        :param inputs: List of tensors, with 'one_hot' as the first tensor
        :return: Output tensors of shape (batch, points, si_units, representation_index)
            with dummy atom values zeroed.
        """
        one_hot, *inputs = inputs
        activated_output = super().call(inputs, **kwargs)
        if not isinstance(activated_output, list):
            activated_output = [activated_output]
        return [DummyAtomMasking()([one_hot, tensor]) for tensor in activated_output]


class MolecularConvolution(MolecularLayer, Convolution):
    """
    Input:
        one_hot (batch, points, depth)
        image (batch, points, points, basis_functions)
        vectors (batch, points, points, 3)
        feature_tensors [(batch, points, features_dim, representation_index), ...]
    Output:
        [(batch, points, si_units, representation_index), ...]
    """

    def __init__(self, *args, **kwargs):
        self.total_required_inputs = 4
        super().__init__(*args, **kwargs)


class MolecularSelfInteraction(MolecularLayer, SelfInteraction):
    """
    Input:
        one_hot (batch, points, depth)
        feature_tensors [(batch, points, features_dim, representation_index), ...]
    Output:
        [(batch, points, si_units, representation_index), ...]
    """

    def __init__(self, *args, **kwargs):
        self.total_required_inputs = 2
        super().__init__(*args, **kwargs)


class MolecularActivation(MolecularLayer, EquivariantActivation):
    """
    Input:
        one_hot (batch, points, depth)
        feature_tensors [(batch, points, features_dim, representation_index), ...]
    Output:
        [(batch, points, si_units, representation_index), ...]
    """

    def __init__(self, *args, **kwargs):
        self.total_required_inputs = 2
        super().__init__(*args, **kwargs)
