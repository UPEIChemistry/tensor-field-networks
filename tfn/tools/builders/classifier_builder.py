import tensorflow as tf
from tensorflow.python.keras.layers import Lambda

from tfn.layers import EquivariantActivation, MolecularConvolution
from tfn.tools.builders import Builder


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


class ClassifierBuilder(Builder, ClassifierMixIn):
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
