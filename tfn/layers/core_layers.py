from keras import backend as K
from keras.activations import relu
from keras.layers import Layer
from keras.initializers import glorot_normal, constant
from . import support_layers
from .. import utils


class RotationallyEquivariantLayer(Layer):

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


class Convolution(RotationallyEquivariantLayer):
    """
    hidden_dim: int. Defaults to input_shape[-1]. Radial fxn is a 2 layer dense-net based off of
    RBF inputs of molecules

    filter_dim: int. Defaults to 1. What the final dimension of the Radial tensor is,
    i.e. [N, N, output_dim] where N is number of points

    activation: func. Defaults to keras.activations.relu. Nonlinearity function to apply to the
    hidden layer of the Radial function.

    weight_initializer: initializer object. Defaults to keras.initializers.glorot_normal.

    bias_initializer: initializer object. Defaults to keras.initializers.constant.

Args specific to Filter1 & Filter2 (Required for Spherical Harmonics):
    dist_matrix: np.ndarray. Distance matrix for the positions of the input system


    :return: HarmonicFilter layer of rotation order 0 and 1
    """
    def __init__(self,
                 image,
                 unit_vectors,
                 hidden_dim=None,
                 filter_dim=1,
                 activation=None,
                 weight_initializer=None,
                 bias_initializer=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.image = image
        self.unit_vectors = unit_vectors
        self.input_dim = int(self.image.shape[-1])
        if hidden_dim is None:
            hidden_dim = self.input_dim
        self.hidden_dim = hidden_dim
        self.filter_dim = filter_dim
        if activation is None:
            activation = relu
        self.activation = activation
        if weight_initializer is None:
            weight_initializer = glorot_normal()
        self.weight_initializer = weight_initializer
        if bias_initializer is None:
            bias_initializer = constant()
        self.bias_initializer = bias_initializer
        self.output_tensors = []

    @utils.wrap_shape_dict
    def build(self, input_shape):
        # FIXME: What the fuck, 4 nested for loops!!??
        for (key, shapes) in input_shape.items():
            for i, shape in enumerate(shapes):
                # 2 weights/biases for 2 filters, so 4 weight/bias pairs per input
                for n in range(2):  # TODO: Two filters
                    for m in range(2):  # TODO: Two sets of weights per filter
                        wname = 'ConvKernel_RO{}_I{}_F{}_W{}'.format(key, i, n, m)
                        bname = 'ConvBias_RO{}_I{}_F{}_W{}'.format(key, i, n, m)
                        if (m + 1) == 2:  # Weight/Bias 2
                            wshape = (self.filter_dim, self.hidden_dim)
                            bshape = (self.filter_dim,)
                        else:  # Weight/Bias 1
                            wshape = (self.hidden_dim, self.input_dim)
                            bshape = (self.hidden_dim, )
                        self.add_weight_to_nested_dict(['kernel', key, i, n, m],
                                                       name=wname,
                                                       shape=wshape,
                                                       initializer=self.weight_initializer)
                        self.add_weight_to_nested_dict(['bias', key, i, n, m],
                                                       name=bname,
                                                       shape=bshape,
                                                       initializer=self.bias_initializer)
        self.built = True

    def compute_output_shape(self, input_shape):
        output_shapes = [
            tuple([
                dim.value for dim in tensor.shape
            ]) for tensor in self.output_tensors
        ]
        return output_shapes

    @utils.inputs_to_dict
    def call(self, inputs, **kwargs):
        """Layer for computing rotationally-equivariant convolution on a set of input features.

        :param inputs: Flattened list of Tensors representing features.
        :return: List of output tensors
        """
        output_tensors = []
        for key, tensors in inputs.items():
            for i, tensor in enumerate(tensors):
                # TODO: If adding support for more filters, add them here!
                f0, f1 = self.get_filters(
                    kernel_dict=self.weight_dict['kernel'][key][i],
                    bias_dict=self.weight_dict['bias'][key][i]
                )
                f0_output = support_layers.EquivariantCombination()([tensor, f0])
                if not isinstance(f0_output, list):
                    f0_output = [f0_output]
                f1_output = support_layers.EquivariantCombination()([tensor, f1])
                if not isinstance(f1_output, list):
                    f1_output = [f1_output]
                output_tensors.extend(f0_output)
                output_tensors.extend(f1_output)

        # FIXME: I feel like this isn't such a great idea...
        self.output_tensors = output_tensors
        return output_tensors

    def get_filters(self, kernel_dict, bias_dict, **kwargs):
        image = kwargs.pop('image', self.image)
        unit_vectors = kwargs.pop('unit_vectors', self.unit_vectors)
        activation = kwargs.pop('activation', self.activation)
        return support_layers.Filter(activation, kernel_dict, bias_dict)(
            [image, unit_vectors]
        )


class Concatenation(Layer):
    """
    Layer for concatenating tensors of the same rotation order along a provided axis, the default being the channels
    axis.

    :param axis: int. Axis that tensors are concatenated across.
    """
    def __init__(self,
                 axis=-2,
                 **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    @utils.inputs_to_dict
    def call(self, inputs, **kwargs):
        output_tensors = []
        for key, tensors in inputs.items():
            output_tensors.append(K.concatenate(tensors, axis=self.axis))
        return output_tensors


class SelfInteraction(RotationallyEquivariantLayer):

    def __init__(self,
                 output_dim,
                 use_bias=True,
                 weight_initializer=None,
                 bias_initializer=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.use_bias = use_bias
        if weight_initializer is None:
            weight_initializer = glorot_normal()
        self.weight_initializer = weight_initializer
        if bias_initializer is None:
            bias_initializer = constant()
        self.bias_initializer = bias_initializer
        self.weight_dict = {}

    @utils.wrap_shape_dict
    def build(self, input_shape):
        for (key, shapes) in input_shape.items():
            for i, shape in enumerate(shapes):
                wname = 'SIKernel_RO{}_I{}'.format(str(key), str(i))
                wshape = (self.output_dim, shape[-2])
                bname = 'SIBias_RO{}_I{}'.format(str(key), str(i))
                bshape = (self.output_dim, )
                self.add_weight_to_nested_dict(['kernel', key, i],
                                               name=wname,
                                               shape=wshape,
                                               initializer=self.weight_initializer
                                               )
                if self.use_bias and key == 0:
                    self.add_weight_to_nested_dict(['bias', key, i],
                                                   name=bname,
                                                   shape=bshape,
                                                   initializer=self.bias_initializer
                                                   )
        self.built = True

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        return [(shape[0], shape[1], self.output_dim, shape[3]) for shape in input_shape]

    @utils.inputs_to_dict
    def call(self, inputs, **kwargs):
        """
        Layer for mixing across channels, also used for increasing dimensions.
        :param inputs: Flat list of input tensors.
        :return: List of output tensors of various rotation orders
        """
        output_tensors = []
        for key, tensors in inputs.items():
            if key == 0:
                for i, tensor in enumerate(tensors):
                    w = self.weight_dict['kernel'][key][i]
                    b = self.weight_dict['bias'][key][i]
                    output_tensors.append(self.self_interaction(tensor, w, b))
            elif key == 1:
                for i, tensor in enumerate(tensors):
                    w = self.weight_dict['kernel'][key][i]
                    output_tensors.append(self.self_interaction(tensor, w))

        return output_tensors

    @staticmethod
    def self_interaction(tensor, w, b=0):
        # a = atoms, f = filters, i = 2l + 1, g = output_dim
        ein_sum = support_layers.EinSum('afi,gf->aig')([tensor, w]) + b
        return K.permute_dimensions(ein_sum, pattern=[0, 2, 1])  # FIXME: May cause issues with batch dim...


class Nonlinearity(RotationallyEquivariantLayer):

    def __init__(self,
                 activation=None,
                 bias_initializer=None,
                 **kwargs):
        super().__init__(**kwargs)
        if activation is None:
            activation = utils.shifted_softplus
        self.activation = activation
        if bias_initializer is None:
            bias_initializer = constant()
        self.bias_initializer = bias_initializer
        self.weight_dict = {}

    @utils.wrap_shape_dict
    def build(self, input_shape):
        for key, shapes in input_shape.items():
            if key != 0:
                for i, shape in enumerate(shapes):
                    name = 'NLBias_RO{}_I{}'.format(key, i)
                    shape = [shape[-2]]
                    self.add_weight_to_nested_dict([key, i],
                                                   name,
                                                   shape,
                                                   initializer=self.bias_initializer
                                                   )

    @utils.inputs_to_dict
    def call(self, inputs, **kwargs):
        output_tensors = []
        for key, tensors in inputs.items():
            if key == 0:
                for i, tensor in enumerate(tensors):
                    output_tensors.append(self.activation(tensor))
            else:
                for i, tensor in enumerate(tensors):
                    b = self.weight_dict[key][i]
                    norm = support_layers.Normalize(keepdims=False)(tensor)
                    a = self.activation(K.bias_add(norm, b))
                    factor = a / norm
                    output_tensors.append(tensor * (K.expand_dims(factor, axis=-1)))

        return output_tensors
