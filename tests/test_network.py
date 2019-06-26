from atomic_images.layers import DistanceMatrix, GaussianBasis, OneHot
from tfn.layers import UnitVectors, SelfInteraction, Convolution, Concatenation, Nonlinearity
from keras import backend as K, optimizers
from keras.models import Model
from keras.layers import Input, Lambda, Reshape


class TestNetwork:

    @staticmethod
    def vector_loss(y_true, y_pred):
        """

        :param y_true: Shape (mols, atoms, 3)
        :param y_pred: Shape (mols, atoms, 3)
        :return: Loss in shape (mols,). Keras averages across batches for the final reported loss value for an epoch
        """

        return K.mean(K.sum(K.square(y_pred - y_true), axis=-1), axis=-1)

    def test_network(self, random_data):

        cart, atom_num, energy = random_data
        r = Input(shape=cart.shape, name='Cartesians', dtype='float32')
        z = Input(shape=atom_num.shape, name='AtomicNums', dtype='int32')
        one_hot = OneHot(10)(z)
        dist = DistanceMatrix()(r)
        image = GaussianBasis(**{
            'width': 0.2,
            'spacing': 0.2,
            'min_value': -1.0,
            'max_value': 15.0
        })(dist)
        unit_vector = UnitVectors()(r)

        one_hot_reshape = Reshape((-1, 10, 1))(one_hot)
        embedding = SelfInteraction(output_dim=15)(one_hot_reshape)
        conv_output = Convolution(dist_matrix=image, unit_vectors=unit_vector)([embedding])
        cat_output = Concatenation()(conv_output)
        si_output = SelfInteraction(output_dim=1)(cat_output)
        predicted_atomic_e = Nonlinearity()(si_output[0])
        predicted_mol_e = Lambda(K.sum)(predicted_atomic_e)

        model = Model(inputs=[r, z], outputs=predicted_mol_e)
        model.compile(
            optimizer=optimizers.get({
                'class_name': 'adam',
                'config': {
                    'lr': 1e-3
                }
            }),
            loss=self.vector_loss
        )
        model.fit(
            x=[cart, atom_num],
            y=energy,
            epochs=3
        )
