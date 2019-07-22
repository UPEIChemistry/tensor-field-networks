import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as K
from tfn.layers import Convolution, SelfInteraction
from tfn.blocks import PreprocessingBlock
from tensorflow.python.keras.models import Model


class MolecularEnergyPredictor(Model):
    def __init__(self,
                 max_z=5,
                 gaussian_config=None,
                 radial_factory=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_z = max_z
        if gaussian_config is None:
            gaussian_config = {'width': 0.2, 'spacing': 0.2, 'min_value': -1.0, 'max_value': 15.0}
        self.gaussian_config = gaussian_config
        self.embedding = SelfInteraction(32)
        self.conv1 = Convolution(radial_factory=radial_factory)
        self.conv2 = Convolution(radial_factory=radial_factory)
        self.conv3 = Convolution(radial_factory=radial_factory)
        self.conv4 = Convolution(radial_factory=radial_factory)

    def call(self, inputs, training=None, mask=None):
        r, z = inputs
        one_hot, rbf, vectors = PreprocessingBlock(self.max_z, self.gaussian_config)([r, z])
        one_hot = tf.reshape(one_hot, [-1, self.max_z, 1])
        embedding = self.embedding(one_hot)
        output = self.conv1([rbf, vectors] + embedding)
        output = self.conv2([rbf, vectors] + output)
        output = self.conv3([rbf, vectors] + output)
        output = self.conv4([rbf, vectors] + output)
        output = K.sum(output[0])
        return output

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([1])


def molecule_generator():

    cartesians = np.random.rand(500, 17, 3)
    atomic_nums = np.random.randint(low=0, high=10, size=[500, 17])
    energies = np.random.rand(500, 1)
    while True:
        indices = np.random.permutation(cartesians.shape[0])
        cartesians = cartesians[indices, ...]
        atomic_nums = atomic_nums[indices, ...]
        energies = energies[indices, ...]
        for r, z, e in zip(cartesians, atomic_nums, energies):
            r, z = r[z > 0], z[z > 0]
            yield (
                (r, z),
                e
            )


if __name__ == '__main__':
    model = MolecularEnergyPredictor()
    model.compile('adam', 'mae', run_eagerly=True)
    model.fit_generator(molecule_generator(), steps_per_epoch=500, epochs=2)