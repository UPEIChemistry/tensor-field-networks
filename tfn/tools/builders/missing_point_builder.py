from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Lambda
from tfn.layers import MolecularConvolution

from tfn.tools.builders import Builder


class MissingPointBuilder(Builder):
    def get_model_output(self, point_cloud: list, inputs: list):
        output = MolecularConvolution(
            name='dual_layer',
            radial_factory=self.radial_factory,
            si_units=1,  # For molecular energy output
            activation=self.activation,
            dynamic=self.dynamic
        )(point_cloud + inputs)

        # Get Energies (mols, atoms, 1, 1) -> (mols, 1)
        atomic_energies = Lambda(lambda x: K.squeeze(x, axis=-1), name='energy_squeeze')(output[0])
        molecular_energy = Lambda(lambda x: K.sum(x, axis=-2), name='molecular_energy')(atomic_energies)

        # Get Forces (mols, atoms, 1, 3) -> (mols, atoms, 3)
        atomic_forces = Lambda(lambda x: K.squeeze(x, axis=-2), name='force')(output[1])
        missing_atom = Lambda(lambda x: K.mean(x, axis=1), name='missing_atom')(atomic_forces)
        return molecular_energy, missing_atom