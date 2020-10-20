from typing import Union, Tuple

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Add, Input, Lambda
from tfn.layers import (
    DenseRadialFactory,
    MolecularConvolution,
    Preprocessing,
    RadialFactory,
    MolecularSelfInteraction,
    MolecularActivation,
)


class Builder(object):
    def __init__(
        self,
        max_z: int,
        num_points: int,
        name: str = "model",
        mu: Union[int, list] = None,
        sigma: Union[int, list] = None,
        standardize: bool = True,
        trainable_offsets: bool = True,
        embedding_units: int = 32,
        radial_factory: Union[RadialFactory, str] = DenseRadialFactory(),
        num_layers: Union[int, Tuple[int]] = (3,),
        si_units: Union[int, Tuple[int]] = (64, 32, 16),
        output_orders: list = None,
        residual: bool = True,
        activation: str = "ssp",
        dynamic: bool = True,
        **kwargs,
    ):
        self.max_z = max_z
        self.num_points = num_points
        self.name = name
        self.mu = mu
        self.sigma = sigma
        self.standardize = standardize
        self.trainable_offsets = trainable_offsets
        self.embedding_units = embedding_units
        self.radial_factory = radial_factory

        if not isinstance(num_layers, tuple):
            num_layers = (num_layers,)
        self.num_layers = num_layers
        if not isinstance(si_units, tuple):
            si_units = self.tuplize_si_units(si_units, self.num_layers)
        self.si_units = si_units

        self.output_orders = output_orders
        self.residual = residual
        self.activation = activation
        self.dynamic = dynamic
        self.sum_points = kwargs.pop("sum_points", False)

        self.use_scalars = kwargs.pop("use_scalars", True)
        if not self.use_scalars:
            kwargs.setdefault("final_output_orders", [1])
        self.normalize_max = kwargs.pop("normalize_max", None)
        self.normalize_min = kwargs.pop("normalize_min", None)
        self.num_final_si_layers = kwargs.pop("num_final_si_layers", 0)
        self.final_si_units = kwargs.pop("final_si_units", 32)

        self.point_cloud_layer = Preprocessing(
            self.max_z,
            kwargs.pop("basis_config", None),
            kwargs.pop("basis_type", "gaussian"),
            sum_points=self.sum_points,
        )
        self.model = None

    @staticmethod
    def tuplize_si_units(si_units, num_layers):
        return tuple(si_units for _ in range(len(num_layers)))

    def normalize_array(self, array):
        if self.normalize_max and self.normalize_min:
            return (array - self.normalize_max) / (
                self.normalize_min - self.normalize_max
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
            Input([self.num_points,], dtype="int32", name="atomic_nums"),
            Input([self.num_points, 3], dtype="float32", name="cartesians"),
        ]

    def get_layers(self, **kwargs):
        num_layers = kwargs.pop("num_layers", self.num_layers)
        si_units = kwargs.pop("si_units", self.si_units)
        clusters, skips = [], []
        for cluster_num, num_layers_in_cluster in enumerate(num_layers):
            skips.append(
                MolecularConvolution(
                    name=f"cluster{cluster_num}_skip",
                    radial_factory=self.radial_factory,
                    si_units=si_units[cluster_num],
                    output_orders=self.output_orders,
                    activation=self.activation,
                    dynamic=self.dynamic,
                    sum_points=self.sum_points,
                )
            )
            layers = []
            for layer_num in range(num_layers_in_cluster):
                layers.append(
                    MolecularConvolution(
                        name=f"cluster_{cluster_num}/layer_{layer_num}",
                        radial_factory=self.radial_factory,
                        si_units=si_units[cluster_num],
                        output_orders=self.output_orders,
                        activation=self.activation,
                        dynamic=self.dynamic,
                        sum_points=self.sum_points,
                    )
                )
            clusters.append(layers)
        if self.residual:
            return clusters, skips
        else:
            return [layer for cluster in clusters for layer in cluster]

    def get_learned_tensors(self, tensors, point_cloud, clusters=None):
        clusters = clusters or self.get_layers()
        output = tensors
        if self.residual:
            clusters, skips = clusters
            for cluster, skip in zip(clusters, skips):
                shortcut = output
                for layer in cluster:
                    output = layer(point_cloud + output)
                shortcut = skip(point_cloud + shortcut)
                output = [Add()([o, s]) for o, s in zip(output, shortcut)]
        else:
            for layer_num, layer in enumerate(clusters):
                output = layer(point_cloud + output)

        return output

    def make_embedding(self, one_hot, layer: MolecularSelfInteraction = None):
        scalar = Lambda(lambda x: tf.expand_dims(x, axis=-1))(one_hot)
        vector = Lambda(lambda x: tf.tile(x, (1, 1, 1, 3)))(scalar)
        if self.residual:
            pre_embedding = [one_hot, scalar, vector]
        else:
            pre_embedding = [one_hot, scalar]
        layer = layer or MolecularSelfInteraction(self.embedding_units)
        return layer(pre_embedding)

    def get_learned_output(self, inputs: list):
        inputs = [
            inputs[0],
            inputs[-1],
        ]  # General case for a single molecule as input (z, r)
        point_cloud = self.point_cloud_layer(inputs)  # one_hot, rbf, vectors
        embedding = self.make_embedding(one_hot=point_cloud[0])
        output = self.get_learned_tensors(embedding, point_cloud)
        return point_cloud, output

    def get_final_output(self, one_hot: tf.Tensor, inputs: list, output_dim: int = 1):
        output = inputs
        for i in range(self.num_final_si_layers):
            output = MolecularSelfInteraction(self.final_si_units, name=f"si_{i}")(
                [one_hot] + output
            )
            output = MolecularActivation(name=f"ea_{i}")([one_hot] + output)
        output = MolecularSelfInteraction(
            self.final_si_units, name=f"si_{self.num_final_si_layers}"
        )([one_hot] + output)
        output = MolecularActivation(name=f"ea_{self.num_final_si_layers}")(
            [one_hot] + output
        )
        return output

    def get_model_output(self, point_cloud: list, inputs: list):
        raise NotImplementedError

    @property
    def model_config(self):
        if self.model:
            return self.model.to_json()
