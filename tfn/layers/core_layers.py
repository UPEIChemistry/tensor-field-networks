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

    output_dim: int. Defaults to 1. What the final dimension of the Radial tensor is,
    i.e. [N, N, output_dim] where N is number of points

    activation: func. Defaults to keras.activations.relu. Nonlinearity function to apply to the
    hidden layer of the Radial function.

    use_bias: bool. Defaults to True. Determines whether or not to use biases Radial function.

    weight_initializer: initializer object. Defaults to keras.initializers.glorot_normal.

    bias_initializer: initializer object. Defaults to keras.initializers.constant.

Args specific to Filter1 & Filter2 (Required for Spherical Harmonics):
    dist_matrix: np.ndarray. Distance matrix for the positions of the input system


    :return: Filter layer of rotation order 0, 1, or 2
    """
    def __init__(self,
                 dist_matrix=None,
                 unit_vectors=None,
                 hidden_dim=None,
                 output_dim=1,
                 activation=None,
                 use_bias=None,
                 weight_initializer=None,
                 bias_initializer=None,
                 **kwargs):
        super().__init__(**kwargs)
        error_message = ('Arg "{}" must be passed to create Convolution layer. See layer docs for list '
                         + 'of required args.')
        if dist_matrix is None:
            raise ValueError(error_message.format(str(dist_matrix)))
        self.dist_matrix = dist_matrix
        if unit_vectors is None:
            raise ValueError(error_message.format(str(unit_vectors)))
        self.unit_vectors = unit_vectors
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        if activation is None:
            activation = relu
        self.activation = activation
        self.use_bias = use_bias
        if weight_initializer is None:
            weight_initializer = glorot_normal()
        self.weight_initializer = weight_initializer
        if bias_initializer is None:
            bias_initializer = constant()
        self.bias_initializer = bias_initializer

    @utils.wrap_shape_dict
    def build(self, input_shape):
        for (key, shapes) in input_shape.items():
            for i, shape in enumerate(shapes):
                # 4 is for 2 weights/biases for 2 filters, so 4 weight/bias pairs per input
                for n in range(4):  # TODO: Change this number if supporting more filters
                    wname = 'ConvKernel_RO{}_I{}_FW{}'.format(n + 1, key, i)
                    bname = 'ConvBias_RO{}_I{}_FW{}'.format(n + 1, key, i)
                    if self.hidden_dim is None:
                        self.hidden_dim = shape[-1]
                    if (n + 1) % 2 == 0:
                        wshape = (self.hidden_dim, self.output_dim)
                        bshape = (self.output_dim, )
                    else:
                        wshape = (shape[-1], self.hidden_dim)
                        bshape = (self.hidden_dim, )
                    self.add_weight_to_nested_dict(['kernel', key, i, n],
                                                   name=wname,
                                                   shape=wshape,
                                                   initializer=self.weight_initializer
                                                   )
                    if self.use_bias:
                        self.add_weight_to_nested_dict(['bias', key, i, n],
                                                       name=bname,
                                                       shape=bshape,
                                                       initializer=self.bias_initializer
                                                       )
        self.built = True

    @utils.wrap_dict
    def call(self, inputs, **kwargs):
        """Layer for computing rotationally-equivariant convolution on a set of input features.

        :param inputs: Flattened list of Tensors representing features.
        :return: List of output tensors
        """
        output_tensors = []
        for key, tensors in inputs.items():
            for i, tensor in enumerate(tensors):
                output_dim = tensor.shape[-2]
                # TODO: If adding support for more filters, add them here!
                f0, f1 = self.get_filters(
                    output_dim=output_dim,
                    weights=[w for w in self.weight_dict['kernel'][key][i].values()],
                    biases=[b for b in self.weight_dict['bias'][key][i].values()]
                )
                output_tensors.extend(support_layers.EquivariantCombination()([tensor, f0(self.unit_vectors)]))
                output_tensors.extend(support_layers.EquivariantCombination()([tensor, f1(self.unit_vectors)]))

        return output_tensors

    def get_filters(self, weights, biases, **kwargs):
        dist_matrix = kwargs.pop('dist_matrix', self.dist_matrix)
        hidden_dim = kwargs.pop('hidden_dim', self.hidden_dim)
        output_dim = kwargs.pop('output_dim', self.output_dim)
        activation = kwargs.pop('activation', self.activation)
        return support_layers.Filter(
            dist_matrix,
            hidden_dim,
            output_dim,
            activation,
            weights,
            biases
        )


class Concatenation(Layer):

    @utils.wrap_dict
    def call(self, inputs, **kwargs):
        output_tensors = []
        for key, tensors in inputs.items():
            output_tensors.append(K.concatenate(tensors, axis=-2))
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
                                               wname,
                                               wshape,
                                               initializer=self.weight_initializer
                                               )
                if self.use_bias and key == 0:
                    self.add_weight_to_nested_dict(['bias', key, i],
                                                   bname,
                                                   bshape,
                                                   initializer=self.bias_initializer
                                                   )
        self.built = True

    @utils.wrap_dict
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
        return K.permute_dimensions(
            K.transpose(support_layers.EinSum('afi,gf->aig')(tensor, w) + b),
            pattern=[0, 2, 1]
        )


class Nonlinearity(RotationallyEquivariantLayer):

    def __init__(self,
                 activation=None,
                 bias_initializer=None,
                 **kwargs):
        super().__init__(**kwargs)
        if activation is None:
            self.activation = utils.shifted_softplus
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

    @utils.wrap_dict
    def call(self, inputs, **kwargs):
        output_tensors = []
        for key, tensors in inputs.items():
            if key == 0:
                for i, tensor in enumerate(tensors):
                    output_tensors.append(self.activation(tensor))
            else:
                for i, tensor in enumerate(tensors):
                    b = self.weight_dict[key][i]
                    norm = support_layers.Normalize()(tensor)
                    a = self.activation(K.bias_add(norm, b))
                    factor = a / norm
                    output_tensors.append(tensor * (K.expand_dims(factor, axis=-1)))

        return output_tensors
