import tensorflow as tf
from tensorflow.keras.layers import Lambda
from tfn.layers import MolecularConvolution

from tfn.tools.builders import Builder


class MissingPointBuilder(Builder):
    def get_model_output(self, point_cloud: list, inputs: list):
        output = MolecularConvolution(
            name="dual_layer",
            radial_factory=self.radial_factory,
            si_units=1,  # For molecular energy output
            activation=self.activation,
            dynamic=self.dynamic,
        )(point_cloud + inputs)

        # Get Energies (batch, points, 1, 1) -> (batch, 1)
        atomic_energies = Lambda(
            lambda x: tf.squeeze(x, axis=-1), name="energy_squeeze"
        )(output[0])
        molecular_energy = Lambda(
            lambda x: tf.reduce_sum(x, axis=-2), name="molecular_energy"
        )(atomic_energies)

        # Get Forces (batch, points, 1, 3) -> (batch, points, 3)
        atomic_forces = Lambda(lambda x: tf.squeeze(x, axis=-2), name="force")(
            output[1]
        )
        missing_atom = Lambda(lambda x: tf.reduce_mean(x, axis=1), name="missing_atom")(
            atomic_forces
        )
        return molecular_energy, missing_atom
