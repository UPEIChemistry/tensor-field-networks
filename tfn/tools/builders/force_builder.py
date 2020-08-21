import tensorflow as tf
from tensorflow.keras.layers import Lambda

from tfn.layers.atomic_images import Unstandardization
from . import Builder


class ForceBuilder(Builder):
    def get_model_output(self, point_cloud: list, inputs: list):
        tensors = self.get_final_output(point_cloud[0], inputs)
        outputs = []
        if self.use_scalars:
            outputs.append(self.sanitize_molecular_energy(tensors[0], point_cloud[0]))
        outputs.append(self.sanitize_atomic_forces(tensors[-1]))
        return outputs

    def sanitize_molecular_energy(self, energy, one_hot):
        atomic_energies = Lambda(
            lambda x: tf.squeeze(x, axis=-1), name="energy_squeeze"
        )(energy)
        atomic_energies = Unstandardization(
            self.mu[0],
            self.sigma[0],
            trainable=self.trainable_offsets,
            name="atomic_energy",
        )([one_hot, atomic_energies])
        return Lambda(lambda x: tf.reduce_sum(x, axis=-2), name="molecular_energy")(
            atomic_energies
        )

    def sanitize_atomic_forces(self, forces):
        atomic_forces = Lambda(lambda x: tf.squeeze(x, axis=-2), name="force_squeeze")(
            forces
        )
        if self.standardize:
            atomic_forces = Unstandardization(
                self.mu[1],
                self.sigma[1],
                trainable=self.trainable_offsets,
                name="atomic_forces",
            )(atomic_forces)
        return atomic_forces
