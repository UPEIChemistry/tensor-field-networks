import numpy as np
from functools import partial
from typing import Union, Iterable, Callable

import tensorflow as tf
from tensorflow.python.keras import backend as K, Sequential, regularizers
from tensorflow.python.keras.layers import Layer, Dense

from atomic_images.layers import DummyAtomMasking, Unstandardization, GaussianBasis, DistanceMatrix
import tfn.wrappers
from tfn import utils


class RadialFactory(object):
    """
    Default factory class for supplying radial functions to a Convolution layer. Subclass this factory and override its
    'get_radial' method to return custom radial instances/templates.
    """
    def get_radial(self, feature_dim, input_ro=None, filter_ro=None):
        """
        Factory method for obtaining radial functions of a specified architecture, or an instance of a radial function
        (i.e. object which inherits from Layer).

        :param feature_dim: Dimension of the feature tensor being point convolved with the filter produced by this
            radial function. Use to ensure radial function outputs a filter of shape (atoms, feature_dim, filter_ro)
        :param input_ro: Optional. Rotation order of the of the feature tensor point convolved with the filter produced
            by this radial function
        :param filter_ro: Optional. Rotation order of the filter being produced by this radial function.
        :return: Keras Layer object, or subclass of Layer. Must have attr dynamic == True and trainable == True.
        """
        return Sequential([
            Dense(
                32,
                dynamic=True,
                kernel_regularizer=regularizers.l2(0.01),
                bias_regularizer=regularizers.l2(0.01)
            ),
            Dense(
                feature_dim,
                dynamic=True,
                kernel_regularizer=regularizers.l2(0.01),
                bias_regularizer=regularizers.l2(0.01)
            )
        ])


class Convolution(Layer):
    """
    Rotationally equivariant convolution operation to be applied to feature tensor(s) of a 3D point-cloud of either
    rotation order 0 or 1, no support for rotation order 2 inputs yet. The arg 'inputs' in the call method is a variable
    length list of tensors in the order: [image, vectors, feature_tensor0, feature_tensor1, ...].

    The operation has several steps:
    1) Generation of filters. 'radial_factory', image and vectors and are used to generate filters, which contain
    the majority of learnable parameters of the Convolution layer. The radial_factory is used to create radial objects
    (which inherit from Layer) to produce directionless filters through a TensorDot from a provided discretized
    distance matrix (image) of the 3D point-cloud. These directionless filters are then EinSumed with spherical harmonic
    functions (produced by vectors) to produce HarmonicFilters.

    2) Point convolving HarmonicFilter with feature tensors. There are several types of combinations possible
    depending on the rotation order (RO, ro) of input and filter. With notation input_ro x filter_ro -> output_ro,
    supported combinations include: L x 0 -> L; 0 x 1 -> 1; 1 x 0 -> 1; 1 x 1 -> 0; 1 x 1 -> 1.
    For example, if input to this Convolution layer is the list of feature tensors with shapes (mols, atoms,
    feature_dim, 1) and (mols, atoms, feature_dim, 3) (i.e. one RO0 tensor and one RO1 tensor) then there are 5 total
    combinations for these tensors (2 for RO0, 3 for RO1) and thus 5 resulting tensors.

    3) Concatenation of resultant tensors. In our previous example where we inputted two tensors of shapes (mols, atoms,
    feature_dim, 1), (mols, atoms, feature_dim, 3) and got 5 resulting tensors (2 for RO0, 3 for RO1), we concatenate
    each set of rotation order tensors along their feature_dim axis, which is analogous to the channels dim of typical
    convolutional networks. This converts our 5 tensors to 2 output tensors, one for each rotation order.

    4) Self Interaction across channels. Next, each tensor (1 for each RO) gets a seperate kernel applied of shape
    (filter_dim, si_units), allowing information mixing across the feature_dim, the dimension analogous to the
    channels dim of a typical conv. net. The output of this layer is a list of tensors of shape (mols, atoms, si_units,
    representation_index), where representation_index refers to RO of the tensor.

    5) Equivariant Activation. Activations need to operate on scalar values, so RO higher than 0 must be reduced to
    scalars. This is completed using an l2_norm on the representation_index (the last axis of operand tensors). This
    reduced-to-scalar norm is then funneled through the specified activation, after which it is cast back up to its
    original RO. The list of output tensors returned from this op is the list returned from the Convolution layer. This
    layer returns as many tensors as rotation orders held by all tensors in the network. E.g. if you have a single
    feature tensor RO0 and apply RO0 & RO1 filters to it, you'll have two output tensors, one RO0 and one RO1.

    :param radial_factory: RadialFactory object which returns a 'radial' function (a Keras Layer object). Defaults to
        base RadialFactory which returns radials of the architecture:
        Sequential([Dense(feature_dim, dynamic=True), Dense(feature_dim, dynamic=True)]). There are
        several requirements of this param and the radial returned by it:
        1) radial_factory must inherit from RadialFactory, i.e. it must have a 'get_radial' method.
        2) radial returned by radial_factory must inherit from Layer, it must be learnable (radial.trainable == True),
        and it must be set to only run in eager (i.e. radial.dynamic == True)
        3) If supplying a custom factory object, either ensure the instance/architecture returned by will convolve with
        the associated feature tensors, or use the arg 'feature_dim' to ensure the filter produced by the radial is of
        the appropriate shape.
    :param si_units: int. Defaults to 16. The output tensor(s) of a Convolution layer are of shape
        (mols, atoms, si_units, representation_index). This param is analogous to the number of filters in typical
        convolutional networks.
    :param activation: str or keras.activation. What nonlinearity should be applied to the output of the network
    :param filter_ro: int or sequence of bools. Defaults to 1. If single int is passed, creates filters for each RO
        from [0, filter_ro]. If sequence is passed, then list index refers which RO values to use. E.g. passing
        [False, True] will produce only filters of RO1, not RO0.
    """
    def __init__(self,
                 radial_factory=None,
                 si_units: int = 16,
                 activation: Union[str, Callable] = 'relu',
                 filter_ro: Union[int, Iterable[bool]] = 1,
                 **kwargs):
        super().__init__(dynamic=True, **kwargs)
        if radial_factory is None:
            radial_factory = RadialFactory()
        self.radial_factory = radial_factory
        self.si_units = si_units
        self.activation = activation
        self.filter_ro = filter_ro

        self._filters = {}
        self._si_layer = None
        self._activation_layer = None

    def build(self, input_shape):
        # Assign static block layers
        self._si_layer = SelfInteraction(self.si_units)
        self._activation_layer = EquivariantActivation(self.activation)

        # Validation and parameter prepping
        if isinstance(self.filter_ro, int):
            filter_orders = range(self.filter_ro + 1)
        else:
            filter_orders = [i for i, f in zip([0, 1], self.filter_ro) if f]
        if not isinstance(self.radial_factory.get_radial(1, 0, 0), Layer):  # This may be costly, and not required
            raise ValueError(
                'passed radial_factory returned radial of type: {}, '
                'while radial must inherit from "Layer"'.format(type(self.radial_factory()).__name__)
            )
        if len(input_shape) < 3:
            raise ValueError('Inputs must contain tensors: "image", "vectors", and feature tensors '
                             'of the 3D point-cloud')
        input_shape = input_shape[2:]
        # Assign radials to filters, and filters to self._filters dict
        self._filters = {
            self.get_tensor_ro(shape): [
                HarmonicFilter(
                    self.radial_factory.get_radial(
                        shape[-2],
                        input_ro=self.get_tensor_ro(shape),
                        filter_ro=filter_ro
                    ),
                    filter_ro=filter_ro
                )
                for filter_ro in filter_orders
            ] for shape in input_shape
        }

    def call(self, inputs, **kwargs):
        """
        :param inputs: List of tensors in the order: image, vectors, feature tensors
        :return: Output tensors of shape (mols, atoms, si_units, representation_index)
        """
        if len(inputs) < 3:
            raise ValueError('Inputs must contain tensors: "image", "vectors", and a list of features tensors.')
        image, vectors, *feature_tensors = inputs
        conv_outputs = self.point_convolution(feature_tensors, image, vectors)
        concat_outputs = self.concatenation(conv_outputs)
        si_outputs = self.self_interaction(concat_outputs)
        return self.equivariant_activation(si_outputs)

    def point_convolution(self, inputs, image, vectors):
        output_tensors = []
        if not isinstance(inputs, list):
            inputs = [inputs]
        for tensor in inputs:
            input_ro = self.get_tensor_ro(tensor)
            filter_outputs = [f([image, vectors]) for f in self._filters[input_ro]]
            cg = self.cg_coefficient(tensor.shape[-1], axis=-2)
            output_tensors.append(tf.einsum('ijk,mabfj,mbfk->mafi', cg, filter_outputs[0], tensor))
            if input_ro == 0:
                # 0 x 1 -> 1
                cg = self.cg_coefficient(3, axis=-1)
                output_tensors.append(tf.einsum('ijk,mabfj,mbfk->mafi', cg, filter_outputs[1], tensor))
            if input_ro == 1:
                # 1 x 1 -> 0
                cg = self.cg_coefficient(3, axis=0)
                output_tensors.append(tf.einsum('ijk,mabfj,mbfk->mafi', cg, filter_outputs[1], tensor))
                # 1 x 1 -> 1
                lc_tensor = self.lc_tensor()
                output_tensors.append(tf.einsum('ijk,mabfj,mbfk->mafi', lc_tensor, filter_outputs[1], tensor))

        return output_tensors

    @staticmethod
    @tfn.wrappers.inputs_to_dict
    def concatenation(inputs: list, axis=-2):
        return [K.concatenate(tensors, axis=axis) for tensors in inputs.values()]

    def self_interaction(self, inputs):
        return self._si_layer(inputs)

    def equivariant_activation(self, inputs):
        return self._activation_layer(inputs)

    @staticmethod
    def get_tensor_ro(tensor):
        try:
            return int((tensor.shape[-1] - 1) / 2)
        except AttributeError:
            return int((tensor[-1] - 1) / 2)

    @staticmethod
    def cg_coefficient(size, axis, dtype=tf.float32):
        """
        Clebsch-Gordan coefficient of varying size and shape.
        """
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


class MolecularConvolution(Convolution):

    def build(self, input_shape):
        if len(input_shape) < 4:
            raise ValueError('MolConvolution layer must be passed tensors: "one_hot", "image", "vectors", and any '
                             'feature tensors associated with the point-cloud')
        _, *input_shape = input_shape
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """
        :param inputs: List of tensors in the order: one_hot, image, vectors, and any feature tensors of the point-cloud
        :return: Output tensors of shape (mols, atoms, si_units, representation_index) with dummy atom values zeroed.
        """
        one_hot, *inputs = inputs
        activated_output = super().call(inputs, **kwargs)
        if not isinstance(activated_output, list):
            activated_output = [activated_output]
        return [
            DummyAtomMasking()([one_hot, tensor]) for tensor in activated_output
        ]


class HarmonicFilter(Layer):
    """
    Layer for generating filters from radial functions.

    :param radial: Callable. The learnable bits of an equivariant filter. Radial can be any tf callable
        (model, layer, op...) that takes the RBF image of shape (atoms, atoms, rbf) as input and combines it
        in some way with weights to return a learned tensor of shape (atoms, atoms, output_dim) that, when combined
        with a tensor derived from a spherical harmonic function aligned with provided unit vectors, returns a filter.
    :param filter_ro: int. What rotation order the filter is.
    """
    def __init__(self,
                 radial,
                 filter_ro=0,
                 **kwargs):
        super().__init__(dynamic=True, **kwargs)
        self.filter_ro = filter_ro
        if not isinstance(radial, Layer):
            raise ValueError('Radial must subclass Layer, but is of type: {}'.format(type(radial).__name__))
        self.radial = radial
        self.filter_ro = filter_ro

    @property
    def trainable_weights(self):
        if self.trainable:
            return self.radial.trainable_weights
        else:
            return []

    def call(self, inputs, **kwargs):
        """Generate the filter based on provided image (and vectors, depending on requested filter rotation order).

        :param inputs: List of input tensors in the order: image, vectors.
        :return: HarmonicFilter tensor of shape: (mols, atoms, filter_dim, representation_index), where filter_dim is
            determined by the radial function.
        """
        image, vectors = inputs
        if self.filter_ro == 0:
            # [mols, N, N, output_dim, 1]
            return K.expand_dims(self.radial(image), axis=-1)
        elif self.filter_ro == 1:
            masked_radial = self.mask_radial(self.radial(image), vectors)
            # [mols, N, N, output_dim, 3]
            return K.expand_dims(vectors, axis=-2) * K.expand_dims(masked_radial, axis=-1)
        elif self.filter_ro == 2:
            masked_radial = self.mask_radial(self.radial(image), vectors)
            # [mols, N, N, output_dim, 5]
            return K.expand_dims(self.l2_spherical_harmonic(vectors), axis=-2) * K.expand_dims(masked_radial, axis=-1)
        else:
            raise ValueError('Unsupported RO passed for filter_ro, only capable of supplying filters of up to and '
                             'including RO2.')

    @staticmethod
    def mask_radial(radial, vectors):
        norm = tf.norm(vectors, axis=-1)
        condition = K.expand_dims(norm < K.epsilon(), axis=-1)
        tile = K.tile(condition, [1, 1, 1, radial.shape[-1]])

        # [N, N, output_dim]
        return tf.where(tile, K.zeros_like(radial), radial)

    @staticmethod
    def l2_spherical_harmonic(tensor):
        """
        Spherical harmonic functions for the L=2 example.

        :param tensor: must be of shape [atoms, atoms, 3]
        :return: tensor. Result of L2 spherical harmonic function applied to input tensor
        """
        x = tensor[:, :, :, 0]
        y = tensor[:, :, :, 1]
        z = tensor[:, :, :, 2]
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
        # if self.activity_regularizer is None:`
        #     self.activity_regularizer = regularizers.l2(0.)`

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

    @tfn.wrappers.shapes_to_dict
    def build(self, input_shape):
        for (key, shapes) in input_shape.items():
            for i, shape in enumerate(shapes):
                self.add_weight_to_nested_dict(
                    [key, i],
                    name='SIKernel_RO{}_I{}'.format(str(key), str(i)),
                    shape=(self.units, shape[-2]),
                    regularizer=self.activity_regularizer
                )
        self.built = True

    @tfn.wrappers.inputs_to_dict
    def call(self, inputs, **kwargs):
        output_tensors = []
        for key, tensors in inputs.items():
            for i, tensor in enumerate(tensors):
                w = self.weight_dict[key][i]
                output_tensors.append(
                    tf.transpose(tf.einsum('mafi,gf->maig', tensor, w), perm=[0, 1, 3, 2])
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
            activation = self.shifted_softplus
        self.activation = activation

    @staticmethod
    def shifted_softplus(x):
        return tf.math.log(0.5 * tf.exp(x) + 0.5)

    @tfn.wrappers.shapes_to_dict
    def build(self, input_shape):
        for (key, shapes) in input_shape.items():
            for i, shape in enumerate(shapes):
                self.add_weight_to_nested_dict(
                    [key, i],
                    name='RTSBias_RO{}_I{}'.format(str(key), str(i)),
                    shape=(shape[-2],),
                    regularizer=self.activity_regularizer
                )
        self.built = True

    @tfn.wrappers.inputs_to_dict
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


class Preprocessing(Layer):

    def __init__(self,
                 max_z,
                 gaussian_config=None,
                 **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.max_z = max_z
        if gaussian_config is None:
            gaussian_config = {
                'width': 0.2, 'spacing': 0.2, 'min_value': -1.0, 'max_value': 15.0
            }
        self.gaussian_config = gaussian_config
        self.one_hot = partial(tf.one_hot, depth=self.max_z)

    def call(self, inputs, **kwargs):
        """
        Convert cartesians and atomic_nums into required tensors
        :param inputs: list. cartesian coordinates and atomic nums, in that order
        :return: list. one_hot, rbf, and unit_vectors tensors in that order.
        """
        r, z = inputs
        return [
            self.one_hot(z),
            GaussianBasis(**self.gaussian_config)(DistanceMatrix()(r)),
            UnitVectors()(r)
        ]


class UnitVectors(Layer):

    def __init__(self,
                 axis=-1,
                 keepdims=True,
                 **kwargs):
        super(UnitVectors, self).__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, inputs, **kwargs):
        """

        :param inputs: cartesian tensors of shape (batch, points, 3)
        :param kwargs:
        :return:
        """
        i = K.expand_dims(inputs, axis=-2)
        j = K.expand_dims(inputs, axis=-3)
        v = i - j
        den = utils.norm_with_epsilon(v, self.axis, self.keepdims)
        return v / den


