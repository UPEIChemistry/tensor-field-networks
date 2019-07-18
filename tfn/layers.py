from collections import Iterable
from typing import Union, Iterable as iterable, Callable

import tensorflow as tf
from tensorflow.python.keras import backend as K, Sequential
from tensorflow.python.keras.engine import Layer
from tensorflow.python.keras.layers import Layer, Dense
from tfn import utils
import numpy as np


class RadialFactory(object):
    """
    Default factory class for supplying radial functions to a Convolution layer. Subclass this factory and override its
    'get_radial' method to return custom radial instances/templates.
    """
    def get_radial(self, feature_dim, input_ro=None, filter_ro=None):
        """

        :param feature_dim: Dimension of the feature tensor being point convolved with the filter produced by this
            radial function. Use to ensure radial function outputs a filter of shape (atoms, feature_dim, filter_ro)
        :param input_ro: Optional. Rotation order of the of the feature tensor point convolved with the filter produced
            by this radial function
        :param filter_ro: Optional. Rotation order of the filter being produced by this radial function.
        :return: Keras Layer object, or subclass of Layer. Must have attr dynamic == True and trainable == True.
        """
        return Sequential([
            Dense(32, dynamic=True),
            Dense(feature_dim, dynamic=True)
        ])


class Convolution(Layer):
    """
    Rotationally equivariant convolution operation to be applied to feature tensor(s) of a 3D point-cloud of either
    rotation order 0 or 1, no support for rotation order 2 inputs yet. The operation has several steps:

    1) Generation of filters. Args 'image', 'vectors', and 'radial_factory' are used to generate filters, which contain
    the majority of learnable parameters of the Convolution layer. The radial_factory is used to create radial objects
    (which inherit from Layer) to produce directionless filters through a TensorDot from a provided discretized
    distance matrix (image) of the 3D point-cloud. These directionless filters are then EinSumed with spherical harmonic
    functions (produced by the unit vectors of the difference matrix of the point-cloud) to produce HarmonicFilters.

    2) Point convolving HarmonicFilter with feature tensors. There are several types of combinations possible
    depending on the rotation order (RO, ro) of input and filter. With notation input_ro x filter_ro -> output_ro,
    supported
    combinations include: L x 0 -> L; 0 x 1 -> 1; 1 x 0 -> 1; 1 x 1 -> 0; 1 x 1 -> 1. For example, if input to this
    Convolution layer is the list of feature tensors with shapes [(atoms, feature_dim, 1), (atoms, feature_dim, 3)]
    (i.e. one RO0 tensor and one RO1 tensor) then there are 5 total combinations for these tensors
    (2 for RO0, 3 for RO1) and thus 5 resulting tensors.

    3) Concatenation of resultant Einsum tensors. In our previous example where we inputted two tensors of shapes
    atoms, feature_dim, 1), (atoms, feature_dim, 3) and got 5 resulting tensors (2 for RO0, 3 for RO1), we concatenate
    each set of rotation order tensors along their feature_dim axis, which is analogous to the channels dim of typical
    convolutional networks. This converts our 5 tensors to 2 output tensors, one for each rotation order.

    4) Self Interaction across channels. Next, each tensor (1 for each RO) gets a seperate kernel applied, allowing
    information mixing across the feature_dim, the dimension analogous to the channels dim of a typical conv. net.

    5) Equivariant Activation. Activations need to operate on scalar values, so RO higher than 0 must be reduced to
    scalars. This is completed using an l2_norm on the
    representation_index (the last axis of operand tensors). This reduced-to-scalar norm is then funneled through the
    specified activation, after which it is cast back up to its original RO. The list of output tensors returned from
    this op is the list returned from the Convolution layer. This layer returns as many tensors as rotation orders held
    by all tensors in the network. E.g. if you have a single input tensor RO0 and apply RO0 & RO1 filters to it, you'll
    have two output tensors, one RO0 and one RO1.

    :param image: np.ndarray. Discretized 3D point-cloud distance matrix. Of shape (points, points, units) where
        units refers to the number of bins used to represent continuous space. In an molecular setting, points are
        atoms, and units are number of radial basis functions which activate at a particular interaction distance
        between two atoms.
    :param vectors: np.ndarray. Unit vectors of a 3D-point cloud difference matrix. Of shape (points, points, 3).
    :param radial_factory: RadialFactory object which returns a 'radial' function (a Keras Layer object). Defaults to
        base RadialFactory which returns radials of the architecture:
        Sequential([Dense(feature_dim, dynamic=True), Dense(feature_dim, dynamic=True)]). There are
        several requirements of this param and the radial returned by it:
        1) radial_factory must inherit from RadialFactory, i.e. it must have a 'get_radial' method.
        2) radial must inherit from Layer, it must be learnable (radial.trainable == True), and it must be set to
        only run in eager (i.e. radial.dynamic == True)
        3) If supplying a custom factory object, either ensure the instance/architecture returned by will convolve with
        the associated feature tensors, or use the arg 'feature_dim' to ensure the filter produced by the radial is of
        the appropriate shape.
    :param si_units: int. Defaults to 16. The output tensor(s) of a Convolution layer are of shape
        (atoms, si_units, representation_index). This param is analogous to the channels dim of typical
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
                 filter_ro: Union[int, iterable[bool]] = 1,
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
        :return: Feature tensors convolved with filters of shape (points, si_units, representation_index)
        """
        if len(inputs) < 3:
            raise ValueError('Inputs must contain tensors: "image", "vectors", and a list of features tensors.')
        image, vectors, feature_tensors = inputs[0], inputs[1], inputs[2:]
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
        else:
            raise ValueError('Unsupported RO passed for filter_ro, only capable of supplying filters of up to and '
                             'including RO2.')

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


class DistanceMatrix(Layer):
    """
    Distance matrix layer

    Expands Cartesian coordinates into a distance matrix.

    Input: coordinates (..., atoms, 3)
    Output: distance matrix (..., atoms, atoms)
    """
    def call(self, inputs, **kwargs):
        # `inputs` should be Cartesian coordinates of shape
        #    (..., atoms, 3)
        v1 = K.expand_dims(inputs, axis=-2)
        v2 = K.expand_dims(inputs, axis=-3)

        sum_squares = K.sum(K.square(v2 - v1), axis=-1)
        sqrt = K.sqrt(sum_squares + K.epsilon())
        K.switch(sqrt >= K.epsilon(), sqrt, K.zeros_like(sqrt))
        return sqrt

    def compute_output_shape(self, positions_shape):
        return positions_shape[0:-2] + (positions_shape[-2], positions_shape[-2])


class KernelBasis(Layer):
    """Expand tensor using kernel of width=width, spacing=spacing,
    starting at min_value ending at max_value (inclusive if endpoint=True).

    Input: tensor (batch, atoms, [atoms, [atoms...])
    Output: tensor expanded into kernel basis set (batch, atoms, [atoms, [atoms...]], n_gaussians)

    Args:
        min_value (float, optional): minimum value
        max_value (float, optional): maximum value (non-inclusive)
        width (float, optional): width of kernel functions
        spacing (float, optional): spacing between kernel functions
        self_thresh (float, optional): value below which a distance is
            considered to be a self interaction (i.e. zero)
        include_self_interactions (bool, optional): whether or not to include
            self-interactions (i.e. distance is zero)
    """
    def __init__(self,
                 min_value=-1,
                 max_value=9,
                 width=0.2,
                 spacing=0.2,
                 self_thresh=1e-5,
                 include_self_interactions=True,
                 endpoint=False,
                 **kwargs):
        super(KernelBasis, self).__init__(**kwargs)
        self._n_centers = int(np.ceil((max_value - min_value) / spacing))
        self.min_value = min_value
        self.max_value = max_value
        self.spacing = spacing
        self.width = width
        self.self_thresh = self_thresh
        self.include_self_interactions = include_self_interactions
        self.endpoint = endpoint

    def call(self, inputs, **kwargs):
        in_tensor = K.expand_dims(inputs, -1)
        mu = tf.linspace(self.min_value, self.max_value, self._n_centers)

        mu_prefix_shape = tuple([1 for _ in range(len(tf.shape(in_tensor)) - 1)])
        mu = K.reshape(mu, mu_prefix_shape + (-1,))
        mu = tf.cast(mu, tf.float64)
        values = self.kernel_func(in_tensor, mu)

        if not self.include_self_interactions:
            mask = K.cast(in_tensor >= self.self_thresh, K.floatx())
            values *= mask

        return values

    def kernel_func(self, inputs, centres):
        raise NotImplementedError

    def compute_output_shape(self, in_tensor_shape):
        return in_tensor_shape + (self._n_centers,)


class GaussianBasis(KernelBasis):
    """Expand distance matrix into Gaussians of width=width, spacing=spacing,
    starting at min_value ending at max_value (inclusive if endpoint=True).

        -(x - u)^2
    exp(----------)
        2 * w^2

    where: u is linspace(min_value, max_value, ceil((max_value - min_value) / width))
           w is width

    Input: distance_matrix (batch, atoms, atoms)
    Output: distance_matrix expanded into Gaussian basis set (batch, atoms, atoms, n_centres)
    """
    def kernel_func(self, inputs, centres):
        gamma = -0.5 / (self.width ** 2)
        return tf.exp(gamma * tf.square(inputs - centres))


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

        :param inputs: cartesian tensors of shape (points, 3)
        :param kwargs:
        :return:
        """
        i = K.expand_dims(inputs, axis=-2)
        j = K.expand_dims(inputs, axis=-3)
        v = i - j
        den = K.sqrt(
            K.maximum(K.sum(K.square(v), axis=self.axis, keepdims=self.keepdims), K.epsilon())
        )
        return v / den
