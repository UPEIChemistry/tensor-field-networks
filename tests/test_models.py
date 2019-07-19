import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from tfn.layers import Convolution, SelfInteraction
from tfn.blocks import PreprocessingBlock


class TestEnergyModels:
    def test_default_conv_model_predict_atomic_energies(self, random_cartesians_and_z):
        class MyModel(Model):
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
                r, z = inputs  # (mols, atoms, 3) and (mols, atoms)
                # Slice r, z for single mol
                one_hot, rbf, vectors = PreprocessingBlock(self.max_z, self.gaussian_config)([r, z])
                one_hot = tf.reshape(one_hot, [-1, self.max_z, 1])
                embedding = self.embedding(one_hot)
                output = self.conv1([rbf, vectors] + embedding)
                output = self.conv2([rbf, vectors] + output)
                output = self.conv3([rbf, vectors] + output)
                output = self.conv4([rbf, vectors] + output)

                output = K.sum(output[0], axis=-2)
                return output

            def compute_output_shape(self, input_shape):

                return tf.TensorShape([10, 1])

        cartesians = np.random.rand(10, 3).astype('float32')
        atomic_nums = np.random.randint(5, size=(10, 1))
        e = np.random.rand(10, 1).astype('float32')
        model = MyModel()
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        model.fit(x=[cartesians, atomic_nums], y=e, epochs=2)
