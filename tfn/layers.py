import json
from functools import partial
from logging import warning
from typing import Iterable, Union

import numpy as np
import tensorflow as tf
from atomic_images.layers import DistanceMatrix, DummyAtomMasking, GaussianBasis, OneHot
from tensorflow.python.keras import Sequential, backend as K, regularizers, Model
from tensorflow.python.keras.layers import Dense, Layer, Lambda
from tensorflow.python.keras.models import model_from_json

from tfn import utils


class RadialFactory(object):
    """
    Default factory class for supplying radial functions to a Convolution layer. Subclass this factory and override its
    `get_radial` method to return custom radial instances/templates. You must also override the `to_json` and
    `from_json` and register any custom `RadialFactory` classes to a unique string in the keras global custom objects
    dict.
    """
    def __init__(self,
                 num_layers: int = 2,
                 units: int = 16,
                 kernel_lambda: float = 0.,
                 bias_lambda: float = 0.,
                 **kwargs):
        self.num_layers = num_layers
        self.units = units
        self.kernel_lambda = kernel_lambda
        self.bias_lambda = bias_lambda

    def get_radial(self, feature_dim, input_order=None, filter_order=None):
        """
        Factory method for obtaining radial functions of a specified architecture, or an instance of a radial function
        (i.e. object which inherits from Layer).

        :param feature_dim: Dimension of the feature tensor being point convolved with the filter produced by this
            radial function. Use to ensure radial function outputs a filter of shape (points, feature_dim, filter_order)
        :param input_order: Optional. Rotation order of the of the feature tensor point convolved with the filter
            produced by this radial function
        :param filter_order: Optional. Rotation order of the filter being produced by this radial function.
        :return: Keras Layer object, or subclass of Layer. Must have attr dynamic == True and trainable == True.
        """
        return Sequential([
            Dense(
                self.units,
                kernel_regularizer=regularizers.l2(self.kernel_lambda),
                bias_regularizer=regularizers.l2(self.bias_lambda),
            )
            for _ in range(self.num_layers)
        ] + [
            Dense(
                feature_dim,
                kernel_regularizer=regularizers.l2(self.kernel_lambda),
                bias_regularizer=regularizers.l2(self.bias_lambda),
            )
        ])

    def to_json(self):
        self.__dict__['type'] = type(self).__name__
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, config):
        return cls(**json.loads(config))


class EquivariantLayer(object):

    @staticmethod
    def get_tensor_ro(tensor):
        """
        Converts represenation_index (i.e. tensor.shape[-1]) to RO integer

        :return: int. RO of tensor
        """
        try:
            return int((tensor.shape[-1] - 1) / 2)
        except AttributeError:
            return int((tensor[-1] - 1) / 2)

    @staticmethod
    def get_representation_index_from_ro(ro):
        """
        Converts from RO to representation_index
        """
        return (ro * 2) + 1


class Convolution(Layer, EquivariantLayer):
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
    depending on the rotation order (RO, ro) of input and filter. With notation input_order x filter_order -> output_ro,
    supported combinations include: L x 0 -> L; 0 x 1 -> 1; 1 x 0 -> 1; 1 x 1 -> 0; 1 x 1 -> 1.
    For example, if input to this Convolution layer is the list of feature tensors with shapes (batch, points,
    feature_dim, 1) and (batch, points, feature_dim, 3) (i.e. one RO0 tensor and one RO1 tensor) then there are 5 total
    combinations for these tensors (2 for RO0, 3 for RO1) and thus 5 resulting tensors.

    3) Concatenation of resultant tensors. In our previous example where we inputted two tensors of shapes (batch, points,
    feature_dim, 1), (batch, points, feature_dim, 3) and got 5 resulting tensors (2 for RO0, 3 for RO1), we concatenate
    each set of rotation order tensors along their feature_dim axis, which is analogous to the channels dim of typical
    convolutional networks. This converts our 5 tensors to 2 output tensors, one for each rotation order.

    4) Self Interaction across channels. Next, each tensor (1 for each RO) gets a seperate kernel applied of shape
    (filter_dim, si_units), allowing information mixing across the feature_dim, the dimension analogous to the
    channels dim of a typical conv. net. The output of this layer is a list of tensors of shape (batch, points, si_units,
    representation_index), where representation_index refers to RO of the tensor.

    5) Equivariant Activation. Activations need to operate on scalar values, so RO higher than 0 must be reduced to
    scalars. This is completed using an l2_norm on the representation_index (the last axis of operand tensors). This
    reduced-to-scalar norm is then funneled through the specified activation, after which it is cast back up to its
    original RO. The list of output tensors returned from this op is the list returned from the Convolution layer. This
    layer returns as many tensors as rotation orders held by all tensors in the network. E.g. if you have a single
    feature tensor RO0 and apply RO0 & RO1 filters to it, you'll have two output tensors, one RO0 and one RO1.

    :param radial_factory: RadialFactory object which returns a 'radial' function (a Keras Layer object). Defaults to
        base RadialFactory which returns radials of the architecture:
        Sequential([Dense(feature_dim), Dense(feature_dim)]). There are
        several requirements of this param and the radial returned by it:
        1) radial_factory must inherit from RadialFactory, i.e. it must have a 'get_radial' method.
        2) radial returned by radial_factory must inherit from Layer, it must be learnable (radial.trainable == True),
        and it must be set to only run in eager (i.e. radial.dynamic == True)
        3) If supplying a custom factory object, either ensure the instance/architecture returned by will convolve with
        the associated feature tensors, or use the arg 'feature_dim' to ensure the filter produced by the radial is of
        the appropriate shape.
    :param si_units: int. Defaults to 16. The output tensor(s) of a Convolution layer are of shape
        (batch, points, si_units, representation_index). This param is analogous to the number of filters in typical
        convolutional networks.
    :param activation: str or keras.activation. What nonlinearity should be applied to the output of the network
    :param filter_order: int or sequence of bools. Defaults to 1. If single int is passed, creates filters for each RO
        from [0, filter_order]. If sequence is passed, then list index refers which RO values to use. E.g. passing
        [False, True] will produce only filters of RO1, not RO0.
    """
    def __init__(self,
                 radial_factory: Union[RadialFactory, str] = None,
                 si_units: int = 16,
                 activation: str = 'relu',
                 max_filter_order: Union[int, Iterable[bool]] = 1,
                 output_orders: list = None,
                 **kwargs):
        factory_kwargs = kwargs.pop('factory_kwargs', {})
        super().__init__(**kwargs)
        if isinstance(radial_factory, str):
            try:
                config = json.loads(radial_factory)
            except ValueError:
                radial_factory = tf.keras.utils.get_custom_objects()[radial_factory](**factory_kwargs)
            else:
                radial_factory = self._initialize_factory(config)
        elif isinstance(radial_factory, RadialFactory):
            pass
        elif radial_factory is None:
            radial_factory = RadialFactory()
        else:
            raise ValueError('arg `radial_factory` was of type {}, which is not supported. Read layer docs to see '
                             'what types are allowed for `radial_factory`'.format(type(radial_factory).__name__))
        self.radial_factory = radial_factory
        self.si_units = si_units
        self.activation = activation
        self.max_filter_order = max_filter_order
        if not isinstance(output_orders, list) or output_orders is None:
            output_orders = [0, 1]
        self.output_orders = output_orders

        self._filters = {}
        self._si_layer = None
        self._activation_layer = None

    def get_config(self):
        base = super().get_config()
        updates = dict(
            radial_factory=self.radial_factory.to_json(),
            si_units=self.si_units,
            activation=self.activation,
            max_filter_order=self.max_filter_order,
            output_orders=self.output_orders
        )
        return {**base, **updates}

    @staticmethod
    def _initialize_factory(config: dict):
        return tf.keras.utils.get_custom_objects()[config['type']].from_json(json.dumps(config))

    def build(self, input_shape):
        # Assign static block layers
        self._si_layer = SelfInteraction(self.si_units)
        self._activation_layer = EquivariantActivation(self.activation)

        # Validation and parameter prepping
        if isinstance(self.max_filter_order, int):
            filter_orders = list(range(self.max_filter_order + 1))
        else:
            filter_orders = [i for i, f in zip([0, 1], self.max_filter_order) if f]
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
                        input_order=self.get_tensor_ro(shape),
                        filter_order=filter_order
                    ),
                    filter_order=filter_order
                )
                for filter_order in filter_orders if self._possible_coefficient(self.get_tensor_ro(shape), filter_order)
            ] for shape in input_shape
        }

    def call(self, inputs, **kwargs):
        if len(inputs) < 3:
            raise ValueError('Inputs must contain tensors: "image", "vectors", and a list of features tensors.')
        image, vectors, *feature_tensors = inputs
        conv_outputs = self._point_convolution(feature_tensors, image, vectors)
        concat_outputs = self._concatenation(conv_outputs)
        si_outputs = self._self_interaction(concat_outputs)
        return self._equivariant_activation(si_outputs)

    def _possible_coefficient(self, input_order, filter_order, no_coefficients=True):
        """
        Observes if the input and filter can be combined to produced an output that is contained in
        `self.output_orders`.

        :param no_coefficients: bool. Defaults to `True`. Set to `False` to return the CG/LC coefficient tensor instead.
        :return: `bool` or tensor object
        """
        if input_order == 0 and filter_order == 1 and 1 in self.output_orders:
            return no_coefficients or self.cg_coefficient(self.get_representation_index_from_ro(filter_order), axis=-1)
        elif input_order == 1 and filter_order == 1 and 0 in self.output_orders:
            return no_coefficients or self.cg_coefficient(self.get_representation_index_from_ro(filter_order), axis=0)
        elif input_order == 1 and filter_order == 1 and 1 in self.output_orders:
            return no_coefficients or self.lc_tensor()
        elif filter_order == 0 and input_order in self.output_orders:
            return no_coefficients or self.cg_coefficient(self.get_representation_index_from_ro(input_order), axis=-2)
        else:
            return False

    def _point_convolution(self, inputs, image, vectors):
        output_tensors = []
        if not isinstance(inputs, list):
            inputs = [inputs]
        for tensor in inputs:
            input_order = self.get_tensor_ro(tensor)
            for hfilter in [f([image, vectors]) for f in self._filters[input_order]]:
                filter_order = self.get_tensor_ro(hfilter)
                coefficient = self._possible_coefficient(
                    input_order,
                    filter_order,
                    no_coefficients=False
                )
                if coefficient is not False:
                    output_tensors.append(
                        tf.einsum('ijk,mabfj,mbfk->mafi', coefficient, hfilter, tensor)
                    )
                else:
                    warning('Unable to find appropriate combination: {} x {} -> {}, skipping...'.format(
                        input_order, filter_order, self.output_orders
                    ))
                    continue

        if not output_tensors:
            raise ValueError('No possible combinations between inputs and filters were found for requested output '
                             'tensor orders.')
        return output_tensors

    def _concatenation(self, inputs: list, axis=-2):
        nested_inputs = [
            [x for x in inputs if self.get_tensor_ro(x) == ro]
            for ro in (0, 1, 2)
        ]
        nested_inputs = [x for x in nested_inputs if x]
        return [tf.concat(tensors, axis=axis) for tensors in nested_inputs]

    def _self_interaction(self, inputs):
        return self._si_layer(inputs)

    def _equivariant_activation(self, inputs):
        return self._activation_layer(inputs)

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

    def compute_output_shape(self, input_shape):
        rbf, vectors, *features = input_shape
        mols, atoms, *_ = rbf
        return [
            tf.TensorShape([
                mols,
                atoms,
                self.si_units,
                self.get_representation_index_from_ro(ro)
            ]) for ro in self.output_orders
        ]


class MolecularConvolution(Convolution):
    """
    Input:
        one_hot (batch, points, depth)
        image (batch, points, points, basis_functions)
        vectors (batch, points, points, 3)
        feature_tensors [(batch, points, features_dim, representation_index), ...]
    Output:
        [(batch, points, si_units, representation_index), ...]
    """
    def build(self, input_shape):
        if len(input_shape) < 4:
            raise ValueError('MolConvolution layer must be passed tensors: "one_hot", "image", "vectors", and any '
                             'feature tensors associated with the point-cloud')
        _, *input_shape = input_shape
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """
        :param inputs: List of tensors in the order: one_hot, image, vectors, and any feature tensors of the point-cloud
        :return: Output tensors of shape (batch, points, si_units, representation_index) with dummy atom values zeroed.
        """
        one_hot, *inputs = inputs
        activated_output = super().call(inputs, **kwargs)
        if not isinstance(activated_output, list):
            activated_output = [activated_output]
        return [
            DummyAtomMasking()([one_hot, tensor]) for tensor in activated_output
        ]


class HarmonicFilter(Layer, EquivariantLayer):
    """
    Layer for generating filters from radial functions.

    :param radial: Callable. The learnable bits of an equivariant filter. Radial can be any tf callable
        (model, layer, op...) that takes the RBF image of shape (points, points, rbf) as input and combines it
        in some way with weights to return a learned tensor of shape (points, points, output_dim) that, when combined
        with a tensor derived from a spherical harmonic function aligned with provided unit vectors, returns a filter.
    :param filter_order: int. What rotation order the filter is.
    """
    def __init__(self,
                 radial,
                 filter_order=0,
                 **kwargs):
        super().__init__(**kwargs)
        self.filter_order = filter_order
        if not isinstance(radial, Layer):
            raise ValueError('Radial must subclass Layer, but is of type: {}'.format(type(radial).__name__))
        self.radial = radial
        self.filter_order = filter_order

    @property
    def trainable_weights(self):
        """
        Keras walks this list when calculating gradients to apply updates
        """
        if self.trainable:
            return self.radial.trainable_weights
        else:
            return []

    def call(self, inputs, **kwargs):
        """Generate the filter based on provided image (and vectors, depending on requested filter rotation order).

        :param inputs: List of input tensors in the order: image, vectors.
        :return: HarmonicFilter tensor of shape: (batch, points, filter_dim, representation_index), where filter_dim is
            determined by the radial function.
        """
        image, vectors = inputs
        if self.filter_order == 0:
            # (batch, points, points, filter_dim, 1)
            return tf.expand_dims(self.radial(image), axis=-1)
        elif self.filter_order == 1:
            masked_radial = self.mask_radial(self.radial(image), vectors)
            # (batch, points, points, filter_dim, 3)
            return tf.expand_dims(vectors, axis=-2) * tf.expand_dims(masked_radial, axis=-1)
        elif self.filter_order == 2:
            masked_radial = self.mask_radial(self.radial(image), vectors)
            # (batch, points, points, filter_dim, 5)
            return tf.expand_dims(self.l2_spherical_harmonic(vectors), axis=-2) * tf.expand_dims(masked_radial, axis=-1)
        else:
            raise ValueError('Unsupported RO passed for filter_order, only capable of supplying filters of up to and '
                             'including RO2.')

    @staticmethod
    def mask_radial(radial, vectors):
        norm = tf.norm(vectors, axis=-1)
        condition = tf.expand_dims(norm < K.epsilon(), axis=-1)
        tile = tf.tile(condition, [1, 1, 1, radial.shape[-1]])

        # (batch, points, points, output_dim)
        return tf.where(tile, tf.zeros_like(radial), radial)

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
        # return : (points, points, 5)
        output = tf.stack([x * y / r2,
                           y * z / r2,
                           (-tf.square(x) - tf.square(y) + 2. * tf.square(z)) / (2 * np.sqrt(3) * r2),
                           z * x / r2,
                           (tf.square(x) - tf.square(y)) / (2. * r2)],
                          axis=-1)
        return output


class SelfInteraction(Layer, EquivariantLayer):
    """
    Input:
        feature tensors [(batch, points, feature_dim, representation_index), ...]
    Output:
        [(batch, points, si_units, representation_index), ...]

    Glorified Dense Layer for using a weight tensor (units, feature_dim) for independently mixing feature tensors
    along their `feature_dim` axis.

    :param units: int. New feature dimension for output tensors.
    """
    def __init__(self,
                 units: int,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.kernels = None

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        self.kernels = [
            self.add_weight(
                name='sikernel_{}'.format( str(i)),
                shape=(self.units, shape[-2]),
                regularizer=self.activity_regularizer
            ) for i, shape in enumerate(input_shape)
        ]
        self.built = True

    def call(self, inputs, **kwargs):
        output_tensors = []
        if not isinstance(inputs, list):
            inputs = [inputs]
        for i, tensor in enumerate(inputs):
            w = self.kernels[i]
            output_tensors.append(
                tf.transpose(tf.einsum('mafi,gf->maig', tensor, w), perm=[0, 1, 3, 2])
            )
        return output_tensors

    def get_config(self):
        base = super().get_config()
        updates = dict(
            units=self.units
        )
        return {**base, **updates}


class EquivariantActivation(Layer, EquivariantLayer):
    """
    Input:
        feature tensors [(batch, points, feature_dim, representation_index), ...]
    Output:
        [(batch, points, feature_dim, representation_index), ...]

    Applies some potentially non-linear `activation` to the list of input feature tensors.

    :param activation: str, callable. Defaults to shifted_softplus. Activation to apply to input feature tensors.
    """
    def __init__(self,
                 activation: str = 'ssp',
                 **kwargs):
        super().__init__(**kwargs)
        self._activation = activation
        if isinstance(activation, str):
            activation = tf.keras.activations.get(activation)
        else:
            raise ValueError('param `activation` must be a string mapping to a registered keras activation')
        self.activation = activation
        self.biases = None

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        self.biases = [
            self.add_weight(
                name='eabias_{}'.format(str(i)),
                shape=(shape[-2],),
                regularizer=self.activity_regularizer
            ) for i, shape in enumerate(input_shape)
        ]
        self.built = True

    def call(self, inputs, **kwargs):
        output_tensors = []
        if not isinstance(inputs, list):
            inputs = [inputs]
        for i, tensor in enumerate(inputs):
            key = self.get_tensor_ro(tensor)
            b = self.biases[i]
            if key == 0:
                b = tf.expand_dims(tf.expand_dims(b, axis=0), axis=-1)
                output_tensors.append(
                    self.activation(tensor + b)
                )
            elif key == 1:
                norm = utils.norm_with_epsilon(tensor, axis=-1)
                a = self.activation(
                    tf.nn.bias_add(norm, b)
                )
                output_tensors.append(
                    tensor * tf.expand_dims(a / norm, axis=-1)
                )
        return output_tensors

    def get_config(self):
        base = super().get_config()
        updates = dict(
            activation=self._activation
        )
        return {**base, **updates}


class Preprocessing(Layer):
    """
    Input:
        cartesian positions (batch, points, 3)
        point types (batch, point_type)
    Output:
        one_hot (batch, points, depth)
        image (batch, points, points, basis_functions)
        vectors (batch, points, points, 3)

    Convenience layer for obtaining required tensors from cartesian and point type tensors. Defaults to gaussians for
    image basis functions.

    :param max_z: int. Total number of point types + 1 (for 0 type points)
    :param gaussian_config: dict. Contains: 'width' which specifies the size of the gaussian basis functions,
        'spacing' which defines the size of the grid, 'min_value' which specifies the beginning point probed by the
        grid, and 'max_value' which defines the end point of the grid.
    """
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
        self.basis_function = GaussianBasis(**self.gaussian_config)
        self.distance_matrix = DistanceMatrix()
        self.unit_vectors = UnitVectors()

    def call(self, inputs, **kwargs):
        r, z = inputs
        return [
            self.one_hot(z),
            self.basis_function(self.distance_matrix(r)),
            self.unit_vectors(r)
        ]

    def get_config(self):
        base = super().get_config()
        updates = dict(
            max_z=self.max_z,
            gaussian_config=self.gaussian_config
        )
        return {**base, **updates}


class UnitVectors(Layer):
    """
    Input:
        cartesian positions (..., batch, points, 3)
    Output:
        unit vectors between every point in every batch (..., batch, point, point, 3)
    """

    def call(self, inputs, **kwargs):
        i = tf.expand_dims(inputs, axis=-2)
        j = tf.expand_dims(inputs, axis=-3)
        v = i - j
        den = utils.norm_with_epsilon(v, axis=-1, keepdims=True)
        return v / den


def shifted_softplus(x):
    y = K.switch(
        x < 14.,
        K.softplus(K.switch(x < 14., x, K.zeros_like(x))),
        x
    )
    return y - K.log(2.)


tf.keras.utils.get_custom_objects().update({
    'ssp': shifted_softplus,
    'default_radial': RadialFactory
})
