import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Lambda, Add

from atomic_images.layers import DistanceMatrix
from tfn.layers import (
    EquivariantActivation,
    MolecularSelfInteraction,
    MolecularConvolution,
    SelfInteraction,
)

from . import Builder


class DualTrunkBuilder(Builder):
    def get_dual_trunks(self, point_clouds: list):
        embedding_layer = MolecularSelfInteraction(
            self.embedding_units, name="embedding"
        )
        embeddings = [
            embedding_layer(
                [pc[0], Lambda(lambda x: tf.expand_dims(x, axis=-1))(pc[0])]
            )
            for pc in point_clouds
        ]
        layers = self.get_layers()
        inputs = [
            self.get_learned_tensors(e, pc, layers)
            for e, pc in zip(embeddings, point_clouds)
        ]
        return inputs

    def mix_dual_trunks(self, point_cloud: list, inputs: list, output_order: int = 0):
        # Select smaller molecule
        one_hots = [p[0] for p in point_cloud]  # [(batch, points, max_z), ...]
        one_hot = Lambda(
            lambda x: tf.where(tf.reduce_sum(x[0]) > tf.reduce_sum(x[1]), x[1], x[0],),
            name="one_hot_select",
        )(one_hots)
        # Truncate to RO0 outputs
        layer = MolecularConvolution(
            name="truncate_layer",
            radial_factory=self.radial_factory,
            si_units=self.final_si_units,
            activation=self.activation,
            output_orders=[output_order],
            dynamic=self.dynamic,
            sum_points=self.sum_points,
        )
        outputs = [layer(z + x)[0] for x, z in zip(inputs, point_cloud)]
        output = Lambda(lambda x: tf.abs(x[1] - x[0]), name="absolute_difference")(
            outputs
        )
        output = self.get_final_output(one_hot, output)
        return one_hot, output

    def get_final_output(self, one_hot: tf.Tensor, inputs: list, output_dim: int = 1):
        output = inputs
        for _ in range(self.num_final_si_layers):
            output = SelfInteraction(self.final_si_units)(output)
        return SelfInteraction(output_dim)(output)

    def get_model_output(self, point_cloud: list, inputs: list):
        raise NotImplementedError


class TSBuilder(DualTrunkBuilder):
    def __init__(self, *args, **kwargs):
        self.output_distance_matrix = kwargs.pop("output_distance_matrix", True)
        super().__init__(*args, **kwargs)

    def get_inputs(self):
        return [
            Input([self.num_points,], name="atomic_nums", dtype="int32"),
            Input([self.num_points, 3], name="reactant_cartesians", dtype="float32"),
            Input([self.num_points, 3], name="product_cartesians", dtype="float32"),
        ]

    def get_model(self, use_cache=True):
        if self.model is not None and use_cache:
            return self.model
        inputs = self.get_inputs()
        point_cloud, learned_tensors = self.get_learned_output(inputs)
        vectors = self.get_model_output(point_cloud, learned_tensors)
        # mix reactant and product cartesians commutatively
        midpoint = Lambda(lambda x: (x[0] + x[1]) / 2, name="averaged_midpoint")(
            [inputs[1], inputs[2]]
        )
        output = Add(name="ts_cartesians")([midpoint, vectors])  # (batch, points, 3)
        if self.output_distance_matrix:
            output = DistanceMatrix(name="ts_dist_matrix")(
                output
            )  # (batch, points, points)
        self.model = Model(inputs=inputs, outputs=output, name=self.name)
        return self.model

    def get_learned_output(self, inputs: list):
        z, r, p = inputs
        point_clouds = [self.point_cloud_layer([x, z]) for x in (r, p)]
        inputs = self.get_dual_trunks(point_clouds)
        return point_clouds, inputs

    def get_model_output(self, point_cloud: list, inputs: list):
        one_hot, output = self.mix_dual_trunks(point_cloud, inputs, output_order=1)
        vectors = Lambda(lambda x: tf.squeeze(x, axis=-2), name="ts_vectors")(output[0])
        return vectors  # (batch, points, 3)


class ClassifierMixIn(object):
    @staticmethod
    def average_votes(inputs: list):
        output = EquivariantActivation(activation="sigmoid")(inputs)
        output = Lambda(
            lambda x: tf.squeeze(tf.squeeze(x, axis=-1), axis=-1), name="squeeze"
        )(
            output[0]
        )  # (batch, points)
        output = Lambda(
            lambda x: tf.reduce_mean(x, axis=-1, keepdims=True),
            name="molecular_average",
        )(output)
        return output


class TSSiameseClassifierBuilder(DualTrunkBuilder, ClassifierMixIn):
    def get_inputs(self):
        return [
            Input([2, self.num_points,], name="atomic_nums", dtype="int32"),
            Input([2, self.num_points, 3], name="cartesians", dtype="float32"),
        ]

    def get_learned_output(self, inputs: list):
        z, c = inputs
        point_clouds = [
            self.point_cloud_layer([a, b])
            for a, b in zip(
                [c[:, 0], c[:, 1]], [z[:, 0], z[:, 1]]  # Split z, c into 4 arrays
            )
        ]
        inputs = self.get_dual_trunks(point_clouds)
        return point_clouds, inputs

    def get_model_output(self, point_cloud: list, inputs: list):
        one_hot, output = self.mix_dual_trunks(point_cloud, inputs)
        return self.average_votes(output)


class TSClassifierBuilder(Builder, ClassifierMixIn):
    def get_inputs(self):
        return [
            Input([self.num_points,], name="atomic_nums", dtype="int32"),
            Input([self.num_points, 3], name="cartesians", dtype="float32"),
        ]

    def get_learned_output(self, inputs: list):
        inputs = list(
            reversed(inputs)
        )  # Because qm9 orders (c, z), while ts orders (z, c)...
        return super().get_learned_output(inputs)

    def get_model_output(self, point_cloud: list, inputs: list):
        one_hot = point_cloud[0]
        output = MolecularConvolution(
            name="energy_layer",
            radial_factory=self.radial_factory,
            si_units=1,  # For molecular energy output
            activation=self.activation,
            output_orders=[0],
            dynamic=self.dynamic,
            sum_points=self.sum_points,
        )(point_cloud + inputs)
        output = self.get_final_output(point_cloud[0], output)
        return self.average_votes(output)
