from atomic_images.layers import Unstandardization
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers.core import Lambda
from tfn.layers import MolecularConvolution

from . import Builder


class EnergyBuilder(Builder):
    def get_model_output(self, point_cloud: list, inputs: list):
        """
        :return: tf.keras `Model` object. Outputs molecular energy tensor of shape (mols, 1)
        """
        output = MolecularConvolution(
            name='energy_layer',
            radial_factory=self.radial_factory,
            si_units=1,  # For molecular energy output
            activation=self.activation,
            output_orders=[0],
            dynamic=self.dynamic,
            sum_atoms=self.sum_atoms
        )(point_cloud + inputs)
        output = self.get_final_output(point_cloud[0], output)
        atomic_energies = Lambda(lambda x: K.squeeze(x, axis=-1), name='squeeze')(output[0])
        atomic_energies = Unstandardization(
            self.mu,
            self.sigma,
            trainable=self.trainable_offsets,
            name='atomic_energy')(
            [point_cloud[0], atomic_energies]
        )
        return Lambda(lambda x: K.sum(x, axis=-2), name='molecular_energy')(atomic_energies)
