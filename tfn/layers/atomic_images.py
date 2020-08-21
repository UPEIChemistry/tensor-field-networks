import tensorflow as tf
from tensorflow.keras.layers import Layer

from tensorflow.keras import backend as K

import numpy as np


def linspace(*args, **kwargs):
    """
    Keras backend equivalent to numpy and TF's linspace

    Arguments:
        start (float or int): the starting point. If only two values
            are provided, the stop value
        stop (float or int): the stopping point. If only two values
            are provided, the number of points.
        n (int): the number of points to return
    """
    endpoint = kwargs.get("endpoint", True)
    if len(args) == 1:
        raise ValueError("must provide the number of points")
    elif len(args) == 2:
        stop, n = args
        start = 0
    elif len(args) == 3:
        start, stop, n = args
    else:
        raise ValueError("invalid call to linspace")

    range_ = stop - start
    if endpoint:
        step_ = range_ / (n - 1)
    else:
        step_ = range_ / n

    points = tf.range(0, n, dtype=tf.float32)
    points *= step_
    points += start

    return points


class OneHot(Layer):
    """One-hot atomic number layer

    Converts a list of atomic numbers to one-hot vectors

    Input: atomic numbers (batch, atoms)
    Output: one-hot atomic number (batch, atoms, atomic_number)
    """

    def __init__(self, max_atomic_number, **kwargs):
        # Parameters
        self.max_atomic_number = max_atomic_number

        super(OneHot, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        atomic_numbers = inputs
        return tf.one_hot(atomic_numbers, self.max_atomic_number)

    def compute_output_shape(self, input_shapes):
        atomic_numbers = input_shapes
        return tf.TensorShape(list(atomic_numbers) + [self.max_atomic_number])

    def get_config(self):
        base_config = super(OneHot, self).get_config()
        config = {"max_atomic_number": self.max_atomic_number}
        return {**base_config, **config}


class DistanceMatrix(Layer):
    """
    Distance matrix layer

    Expands Cartesian coordinates into a distance matrix.

    Input: coordinates (..., atoms, 3)
    Output: distance matrix (..., atoms, atoms)
    """

    def call(self, inputs, **kwargs):
        positions = inputs
        # `positions` should be Cartesian coordinates of shape
        #    (..., atoms, 3)
        v1 = tf.expand_dims(positions, axis=-2)
        v2 = tf.expand_dims(positions, axis=-3)

        sum_squares = tf.reduce_sum(tf.square(v2 - v1), axis=-1)
        sqrt = tf.sqrt(sum_squares + 1e-7)
        tf.where(sqrt >= 1e-7, sqrt, tf.zeros_like(sqrt))
        return sqrt

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(
            list(input_shape[:-2]) + [input_shape[-2], input_shape[-2]]
        )


#
# Kernel functions
#
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

    def __init__(
        self,
        min_value=-1,
        max_value=9,
        width=0.2,
        spacing=0.2,
        self_thresh=1e-5,
        include_self_interactions=True,
        endpoint=False,
        **kwargs
    ):
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
        in_tensor = tf.expand_dims(inputs, -1)
        mu = linspace(
            self.min_value, self.max_value, self._n_centers, endpoint=self.endpoint
        )

        mu_prefix_shape = tuple([1 for _ in range(len(K.int_shape(in_tensor)) - 1)])
        mu = tf.reshape(mu, mu_prefix_shape + (-1,))
        values = self.kernel_func(in_tensor, mu)

        if not self.include_self_interactions:
            mask = tf.cast(in_tensor >= self.self_thresh, tf.float32)
            values *= mask

        return values

    def kernel_func(self, inputs, centres):
        raise NotImplementedError

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(list(input_shape) + [self._n_centers])

    def get_config(self):
        config = {
            "width": self.width,
            "spacing": self.spacing,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "self_thresh": self.self_thresh,
            "include_self_interactions": self.include_self_interactions,
            "endpoint": self.endpoint,
        }
        base_config = super().get_config()
        return {**base_config, **config}


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

    Args:
        min_value (float, optional): minimum value
        max_value (float, optional): maximum value (non-inclusive)
        width (float, optional): width of Gaussians
        spacing (float, optional): spacing between Gaussians
        self_thresh (float, optional): value below which a distance is
            considered to be a self interaction (i.e. zero)
        include_self_interactions (bool, optional): whether or not to include
            self-interactions (i.e. distance is zero)
                (batch, atoms, atoms, n_gaussians)
    """

    def kernel_func(self, inputs, centres):
        gamma = -0.5 / (self.width ** 2)
        return tf.exp(gamma * tf.square(inputs - centres))


class CosineBasis(KernelBasis):
    """
    Expand distance value into a vector of dampened cosine activations, each element representing
    the activation of a cosine function parameterized by a grid of kappa values, where kappa
    refers to the period size of the cosine function.\n

    f(kappa, x) = cos(kappa * x) * e^(-w * x)

    Where:
        x is our distance value;\n
        w is the width parameter of the dampening;

    Input: distance_matrix (batch, atoms, atoms);\n
    Output: distance_matrix expanded into Cosine basis set (batch, atoms, atoms, n_centres)

    Args:
        min_value (float, optional): minimum value of kappa
        max_value (float, optional): maximum value (non-inclusive) of kappa
        width (float, optional): Parameter for dampening, lower width means the cosine function
            dampens earlier, and only shorter distances are probed. Keep around 0.2
        spacing (float, optional): spacing on the grid of kappa values being used to generate
            cosine basis functions
        self_thresh (float, optional): value below which a distance is
            considered to be a self interaction (i.e. zero)
        include_self_interactions (bool, optional): whether or not to include
            self-interactions (i.e. distance is zero)
                (batch, atoms, atoms, n_gaussians)
    """

    def kernel_func(self, inputs, centres):
        return tf.cos(centres * inputs) * self.cutoff(inputs)

    def cutoff(self, inputs):
        return tf.exp(-self.width * inputs)


class ShiftedCosineBasis(CosineBasis):
    def kernel_func(self, inputs, centres):
        return (0.5 * (tf.cos(centres * inputs) + 1)) * self.cutoff(inputs)


#
# Atom-related functions
#
class AtomicNumberBasis(Layer):
    """Expands Gaussian matrix into the one-hot atomic numbers basis

    Inputs:
        one_hot_numbers  (batch, atoms, max_atomic_number + 1)
        gaussians_matrix  (batch, atoms, atoms, n_gaussians)
    Output:
        gaussians_atom_matrix  (batch, atoms, atoms, n_gaussians, max_atomic_number + 1)
    """

    def __init__(self, zero_dummy_atoms=False, **kwargs):
        kwargs.pop("max_atomic_number", None)  # Backward compatibility
        super(AtomicNumberBasis, self).__init__(**kwargs)
        self.zero_dummy_atoms = zero_dummy_atoms

    def call(self, inputs, **kwargs):
        one_hot_numbers, gaussian_mat = inputs

        gaussian_mat = tf.expand_dims(gaussian_mat, axis=-1)
        if self.zero_dummy_atoms:
            mask = tf.eye(one_hot_numbers.shape[-1], dtype=tf.float32)
            mask[0] = 0
            one_hot_numbers = K.dot(one_hot_numbers, mask)
        one_hot_numbers = tf.expand_dims(one_hot_numbers, axis=1)
        one_hot_numbers = tf.expand_dims(one_hot_numbers, axis=3)
        return gaussian_mat * one_hot_numbers

    def compute_output_shape(self, input_shapes):
        one_hot_numbers_shape, gaussian_mat_shape = input_shapes
        return tf.TensorShape(list(gaussian_mat_shape) + [one_hot_numbers_shape[-1]])

    def get_config(self):
        config = {"zero_dummy_atoms": self.zero_dummy_atoms}
        base_config = super(AtomicNumberBasis, self).get_config()
        return {**base_config, **config}


#
# Normalization-related layers
#
class Unstandardization(Layer):
    """
    Offsets energies by mean and standard deviation (optionally, per-atom)

    `mu` and `sigma` both follow the following:
        If the value is a scalar, apply it equally to all properties
        and all types of atoms

        If the value is a vector, each component corresponds to an
        output property. It is expanded into a matrix where the
        first axis shape is 1. It then follows the matrix rules.

        If the value is a matrix, rows correspond to types of atoms and
        columns correspond to properties.

            If there is only one row, then the row vector applies to every
            type of atom equally.

            If there is one column, then the scalars are applied to every
            property equally.

            If there is a single scalar, then it is treated as a scalar.

    Inputs: the inputs to this layer depend on whether or not mu and sigma
            are given as a single scalar or per atom type.

        If scalar:
            atomic_props  (batch, atoms, energies)
        If per type:
            one_hot_atomic_numbers (batch, atoms, atomic_number)
            atomic_props  (batch, atoms, energies)
    Output: atomic_props  (batch, atoms, energies)

    Attributes:
        mu (float, list, or np.ndarray): the mean values by which
            to offset the inputs to this layer
        sigma (float, list, or np.ndarray): the standard deviation
            values by which to scale the inputs to this layer
    """

    def __init__(
        self, mu, sigma, trainable=False, per_type=False, use_float64=False, **kwargs
    ):
        super(Unstandardization, self).__init__(trainable=trainable, **kwargs)
        self.init_mu = mu
        self.init_sigma = sigma
        self.use_float64 = use_float64

        self.mu = np.asanyarray(self.init_mu)
        self.sigma = np.asanyarray(self.init_sigma)

        self.per_type = len(self.mu.shape) > 0 or per_type

    @staticmethod
    def expand_ones_to_shape(arr, shape):
        if len(arr.shape) == 0:
            arr = arr.reshape((1, 1))
        if 1 in arr.shape:
            tile_shape = tuple(
                shape[i] if arr.shape[i] == 1 else 1 for i in range(len(shape))
            )
            arr = np.tile(arr, tile_shape)
        if arr.shape != shape:
            raise ValueError(
                "the arrays were not of the right shape: "
                "expected %s but was %s" % (shape, arr.shape)
            )
        return arr

    def build(self, input_shapes):
        # If mu is given as a vector, assume it applies to all atoms
        if len(self.mu.shape) == 1:
            self.mu = np.expand_dims(self.mu, axis=0)
        if len(self.sigma.shape) == 1:
            self.sigma = np.expand_dims(self.sigma, axis=0)

        if self.per_type:
            one_hot_atomic_numbers, atomic_props = input_shapes
            w_shape = (one_hot_atomic_numbers[-1], atomic_props[-1])

            self.mu = self.expand_ones_to_shape(self.mu, w_shape)
            self.sigma = self.expand_ones_to_shape(self.sigma, w_shape)
        else:
            w_shape = self.mu.shape

        self.mu = self.add_weight(
            name="mu", shape=w_shape, initializer=lambda x, dtype=self.dtype: self.mu
        )
        self.sigma = self.add_weight(
            name="sigma",
            shape=w_shape,
            initializer=lambda x, dtype=self.dtype: self.sigma,
        )
        super(Unstandardization, self).build(input_shapes)

    def call(self, inputs, **kwargs):
        # `atomic_props` should be of shape (batch, atoms, energies)

        # If mu and sigma are given per atom type, need atomic numbers
        # to know how to apply them. Otherwise, just energies is enough.
        if self.per_type or isinstance(inputs, (list, tuple)):
            self.per_type = True
            one_hot_atomic_numbers, atomic_props = inputs
            atomic_props *= K.dot(one_hot_atomic_numbers, self.sigma)
            atomic_props += K.dot(one_hot_atomic_numbers, self.mu)
        else:
            atomic_props = inputs
            atomic_props *= self.sigma
            atomic_props += self.mu

        return atomic_props

    def compute_output_shape(self, input_shapes):
        if self.per_type or isinstance(input_shapes, list):
            atomic_props = input_shapes[-1]
        else:
            atomic_props = input_shapes
        return atomic_props

    def get_config(self):
        mu = self.init_mu
        if isinstance(mu, (np.ndarray, np.generic)):
            if len(mu.shape) > 0:
                mu = mu.tolist()
            else:
                mu = float(mu)

        sigma = self.init_sigma
        if isinstance(sigma, (np.ndarray, np.generic)):
            if len(sigma.shape) > 0:
                sigma = sigma.tolist()
            else:
                sigma = float(sigma)

        config = {
            "mu": mu,
            "sigma": sigma,
            "per_type": self.per_type,
            "use_float64": self.use_float64,
        }
        base_config = super(Unstandardization, self).get_config()
        return {**base_config, **config}


#
# Dummy atom-related layers
#
class DummyAtomMasking(Layer):
    """
    Masks dummy atoms (atomic number = 0 by default) with zeros

    Inputs: atomic_numbers
                Either or both in this order:
                    atomic_numbers  (batch, atoms)
                or
                    one_hot_atomic_numbers  (batch, atoms, atomic_number)
            value  (batch, atoms, ...)
    Output: value with zeroes for dummy atoms  (batch, atoms, ...)

    Args:
        atom_axes (int or iterable of int): axes to which to apply
            the masking

    Keyword Args:
        dummy_index (int): the index to mask (default: 0)
        invert_mask (bool): if True, zeroes all but the desired index rather
            than zeroeing the desired index
    """

    def __init__(self, atom_axes=1, **kwargs):
        self.invert_mask = kwargs.pop("invert_mask", False)
        self.dummy_index = kwargs.pop("dummy_index", 0)
        super(DummyAtomMasking, self).__init__(**kwargs)
        if isinstance(atom_axes, int):
            atom_axes = [atom_axes]
        elif isinstance(atom_axes, tuple):
            atom_axes = list(atom_axes)
        self.atom_axes = atom_axes

    def call(self, inputs, **kwargs):
        # `value` should be of shape (batch, atoms, ...)
        one_hot_atomic_numbers, value = inputs
        atomic_numbers = tf.argmax(one_hot_atomic_numbers, axis=-1)

        # Form the mask that removes dummy atoms (atomic number = dummy_index)
        if self.invert_mask:
            selection_mask = tf.equal(atomic_numbers, self.dummy_index)
        else:
            selection_mask = tf.not_equal(atomic_numbers, self.dummy_index)
        selection_mask = tf.cast(selection_mask, value.dtype)

        for axis in self.atom_axes:
            mask = selection_mask
            for _ in range(axis - 1):
                mask = tf.expand_dims(mask, axis=1)
            # Add one since tf.int_shape does not return batch dim
            while len(K.int_shape(value)) != len(K.int_shape(mask)):
                mask = tf.expand_dims(mask, axis=-1)

            # Zeros the energies of dummy atoms
            value *= mask
        return value

    def compute_output_shape(self, input_shapes):
        value = input_shapes[-1]
        return value

    def get_config(self):
        config = {
            "atom_axes": self.atom_axes,
            "invert_mask": self.invert_mask,
            "dummy_index": self.dummy_index,
        }
        base_config = super(DummyAtomMasking, self).get_config()
        return {**base_config, **config}


#
# Cutoff functions
#
class CutoffLayer(Layer):
    """Base layer for cutoff functions.

    Applies a cutoff function to the expanded distance matrix

    Inputs:
        distance_matrix (batch, atoms, atoms)
        basis_functions (batch, atoms, atoms, n_centres)
    Output: basis_functions with cutoff function multiplied (batch, atoms, atoms, n_centres)
    """

    def __init__(self, cutoff, **kwargs):
        super(CutoffLayer, self).__init__(**kwargs)
        self.cutoff = cutoff

    def call(self, inputs, **kwargs):
        distance_matrix, basis_functions = inputs

        cutoffs = self.cutoff_function(distance_matrix)
        cutoffs = K.expand_dims(cutoffs, axis=-1)

        return basis_functions * cutoffs

    def cutoff_function(self, distance_matrix):
        """Function responsible for the cutoff. It should also return zeros
        for anything greater than the cutoff.

        Args:
            distance_matrix (Tensor): the distance matrix tensor
        """
        raise NotImplementedError

    def compute_output_shape(self, input_shapes):
        _, basis_functions_shape = input_shapes
        return basis_functions_shape

    def get_config(self):
        config = {"cutoff": self.cutoff}
        base_config = super(CutoffLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CosineCutoff(CutoffLayer):
    """The cosine cutoff originally proposed by Behler et al. for ACSFs.
    """

    def cutoff_function(self, distance_matrix):
        cos_component = 0.5 * (1 + K.cos(np.pi * distance_matrix / self.cutoff))
        return K.switch(
            distance_matrix <= self.cutoff, cos_component, K.zeros_like(distance_matrix)
        )


class TanhCutoff(CutoffLayer):
    """Alternate tanh^3 cutoff function mentioned in some of the ACSF papers.
    """

    def cutoff_function(self, distance_matrix):
        normalization_factor = 1.0 / (K.tanh(1.0) ** 3)
        tanh_component = (K.tanh(1.0 - (distance_matrix / self.cutoff))) ** 3
        return K.switch(
            distance_matrix <= self.cutoff,
            normalization_factor * tanh_component,
            K.zeros_like(distance_matrix),
        )


class LongTanhCutoff(CutoffLayer):
    """Custom tanh cutoff function that keeps symmetry functions relatively unscaled
    longer than the previously proposed tanh function
    """

    def cutoff_function(self, distance_matrix):
        normalization_factor = 1.0 / (K.tanh(float(self.cutoff)) ** 3)
        tanh_component = (K.tanh(self.cutoff - distance_matrix)) ** 3
        return K.switch(
            distance_matrix <= self.cutoff,
            normalization_factor * tanh_component,
            K.zeros_like(distance_matrix),
        )


tf.keras.utils.get_custom_objects().update(
    {
        OneHot.__name__: OneHot,
        DistanceMatrix.__name__: DistanceMatrix,
        KernelBasis.__name__: KernelBasis,
        GaussianBasis.__name__: GaussianBasis,
        AtomicNumberBasis.__name__: AtomicNumberBasis,
        Unstandardization.__name__: Unstandardization,
        DummyAtomMasking.__name__: DummyAtomMasking,
        CutoffLayer.__name__: CutoffLayer,
        CosineCutoff.__name__: CosineCutoff,
        TanhCutoff.__name__: TanhCutoff,
        LongTanhCutoff.__name__: LongTanhCutoff,
    }
)
