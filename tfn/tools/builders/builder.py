from typing import Union, Tuple

import tensorflow as tf
from tensorflow.python.keras import Model, backend as K
from tensorflow.python.keras.layers import Add, Input, Lambda
from tfn.layers import (DenseRadialFactory, MolecularConvolution, Preprocessing, RadialFactory,
                        MolecularSelfInteraction)


class Builder(object):
    def __init__(self,
                 max_z: int,
                 num_atoms: int,
                 name: str = 'model',
                 mu: Union[int, list] = None,
                 sigma: Union[int, list] = None,
                 standardize: bool = True,
                 trainable_offsets: bool = True,
                 embedding_units: int = 32,
                 radial_factory: Union[RadialFactory, str] = DenseRadialFactory(),
                 num_layers: Union[int, Tuple[int]] = (3, ),
                 si_units: Union[int, Tuple[int]] = (64, 32, 16),
                 output_orders: list = None,
                 residual: bool = True,
                 activation: str = 'ssp',
                 dynamic: bool = True,
                 **kwargs):
        self.max_z = max_z
        self.num_atoms = num_atoms
        self.name = name
        self.mu = mu
        self.sigma = sigma
        self.standardize = standardize
        self.trainable_offsets = trainable_offsets
        self.embedding_units = embedding_units
        self.radial_factory = radial_factory

        if not isinstance(num_layers, tuple):
            num_layers = (num_layers, )
        self.num_layers = num_layers
        if not isinstance(si_units, tuple):
            si_units = self.tuplize_si_units(si_units, self.num_layers)
        self.si_units = si_units

        self.output_orders = output_orders
        self.residual = residual
        self.activation = activation
        self.dynamic = dynamic

        self.use_scalars = kwargs.pop('use_scalars', True)
        if not self.use_scalars:
            kwargs.setdefault('final_output_orders', [1])
        self.normalize_max = kwargs.pop('normalize_max', None)
        self.normalize_min = kwargs.pop('normalize_min', None)
        self.num_final_si_layers = kwargs.pop('num_final_si_layers', 0)
        self.final_si_units = kwargs.pop('final_si_units', 32)

        self.point_cloud_layer = Preprocessing(
            self.max_z,
            kwargs.pop('basis_config', None),
            kwargs.pop('basis_type', 'gaussian')
        )
        self.model = None

    @staticmethod
    def tuplize_si_units(si_units, num_layers):
        return tuple(si_units for _ in range(len(num_layers)))

    def normalize_array(self, array):
        if self.normalize_max and self.normalize_min:
            return (
                    (array - self.normalize_max) /
                    (self.normalize_min - self.normalize_max)
            )
        else:
            return array

    def get_model(self, use_cache=True):
        if self.model is not None and use_cache:
            return self.model
        inputs = self.get_inputs()
        point_cloud, learned_tensors = self.get_learned_output(inputs)
        output = self.get_model_output(point_cloud, learned_tensors)
        self.model = Model(inputs=inputs, outputs=output, name=self.name)
        return self.model

    def get_inputs(self):
        return [
            Input([self.num_atoms, 3], dtype='float32', name='cartesians'),
            Input([self.num_atoms, ], dtype='int32', name='atomic_nums')
        ]

    def get_layers(self, **kwargs):
        name = kwargs.pop('name', 'conv')
        num_layers = kwargs.pop('num_layers', self.num_layers)
        si_units = kwargs.pop('si_units', self.si_units)
        layers = []
        for cluster_num, num_layers_in_cluster in enumerate(num_layers):
            for layer_num in range(num_layers_in_cluster):
                layers.append(
                    MolecularConvolution(
                        name='{}_cluster{}_layer{}'.format(name, cluster_num, layer_num),
                        radial_factory=self.radial_factory,
                        si_units=si_units[cluster_num],
                        output_orders=self.output_orders,
                        activation=self.activation,
                        dynamic=self.dynamic
                    )
                )
        return layers

    def get_learned_tensors(self, tensors, point_cloud, layers=None):
        layers = layers or self.get_layers()
        learned_output = tensors
        for layer_num, layer in enumerate(layers):
            if layer_num == 0:
                learned_output = layer(point_cloud + learned_output)
            elif self.residual:
                learned_output = [
                    Add()([x, y]) for x, y in zip(learned_output,
                                                  layer(point_cloud + learned_output))
                ]
            else:
                learned_output = layer(point_cloud + learned_output)
        return learned_output

    def get_learned_output(self, inputs: list):
        inputs = [inputs[0], inputs[-1]]  # General case for a single molecule as input (r, z)
        point_cloud = self.point_cloud_layer(inputs)  # one_hot, rbf, vectors
        expanded_onehot = Lambda(lambda x: K.expand_dims(x, axis=-1))(point_cloud[0])
        embedding = MolecularSelfInteraction(
            self.embedding_units, name='embedding'
        )([point_cloud[0], expanded_onehot])
        output = self.get_learned_tensors(embedding, point_cloud)
        return point_cloud, output

    def get_final_output(self, one_hot: tf.Tensor, inputs: list, output_dim: int = 1):
        output = inputs
        for _ in range(self.num_final_si_layers):
            output = MolecularSelfInteraction(self.final_si_units)([one_hot] + output)
        return MolecularSelfInteraction(output_dim)([one_hot] + output)

    def get_model_output(self, point_cloud: list, inputs: list):
        raise NotImplementedError

    @property
    def model_config(self):
        if self.model:
            return self.model.to_json()
