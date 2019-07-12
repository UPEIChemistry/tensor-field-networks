from atomic_images.layers import DistanceMatrix, GaussianBasis, OneHot
from tfn.layers_v1 import DifferenceMatrix, UnitVectors, SelfInteraction, Convolution, Concatenation, Nonlinearity
from keras import backend as K, optimizers
from keras.callbacks import TensorBoard
from keras.models import Model
from keras.layers import Input, Lambda, Reshape
import numpy as np
from tensorflow.python import debug as tf_debug


def compile_fit_eval_pred(model: Model,
                          inputs,
                          targets,
                          epochs=2,
                          optimizer=None,
                          loss='mean_squared_error',
                          debug=False):
    if optimizer is None:
        optimizer = optimizers.get({
            'class_name': 'adam',
            'config': {
                'lr': 1e-3
            }
        })
    model.compile(
        optimizer=optimizer,
        loss=loss
    )
    if debug:
        sess = K.get_session()
        sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:6064')
        K.set_session(sess)
    model.fit(
        x=inputs,
        y=targets,
        epochs=epochs,
        callbacks=[TensorBoard()]
    )
    loss = model.evaluate(
        x=inputs,
        y=targets
    )
    pred = model.predict(
        x=inputs
    )
    return loss, pred


# Input tensors of rotation order 0 have 2 outputs, while inputs tensors of rotation order 1 have 3 outputs
class TestConvolution:

    def test_ro0_input(self):
        rand_input = np.random.rand(1, 10, 15, 1)
        rand_target = [
          np.random.rand(1, 10, 15, 1), np.random.rand(1, 10, 15, 3)
        ]

        rand_image = np.random.rand(1, 10, 10, 80)
        rand_vectors = np.random.rand(1, 10, 10, 3)

        image = Input(rand_image.shape[1:], name='RandomImageInputForConvolution')
        vectors = Input(rand_vectors.shape[1:], name='RandomVectorsInputForConvolution')
        inp = Input(rand_input.shape[1:], name='RandomInputForConvolution')

        # 0 x 0 -> 0; 0 x 1 -> 1
        conv_output = Convolution(image=image, unit_vectors=vectors, filter_dim=15)(inp)

        model = Model(inputs=[image, vectors, inp], outputs=conv_output)
        loss, pred = compile_fit_eval_pred(
            model,
            inputs=[rand_image, rand_vectors, rand_input],
            targets=rand_target
        )

        assert pred[0].shape == (1, 10, 15, 1) and pred[1].shape == (1, 10, 15, 3)

    def test_ro1_input(self):
        rand_input = np.random.rand(1, 10, 15, 3)
        rand_target = [
          np.random.rand(1, 10, 15, 3), np.random.rand(1, 10, 15, 1), np.random.rand(1, 10, 15, 3)
        ]

        rand_image = np.random.rand(1, 10, 10, 80)
        rand_vectors = np.random.rand(1, 10, 10, 3)

        image = Input(rand_image.shape[1:], name='RandomImageInputForConvolution')
        vectors = Input(rand_vectors.shape[1:], name='RandomVectorsInputForConvolution')
        inp = Input(rand_input.shape[1:], name='RandomInputForConvolution')

        # 1 x 0 -> 1; 1 x 1 -> 0; 1 x 1 -> 1
        conv_output = Convolution(image=image, unit_vectors=vectors, filter_dim=15)(inp)

        model = Model(inputs=[image, vectors, inp], outputs=conv_output)
        loss, pred = compile_fit_eval_pred(
            model,
            inputs=[rand_image, rand_vectors, rand_input],
            targets=rand_target,
            debug=True
        )

        assert pred[0].shape == (1, 10, 15, 3) and pred[1].shape == (1, 10, 15, 1) and pred[2].shape == (1, 10, 15, 3)

    def test_multiple_ro_inputs(self):
        rand_inputs = [
            np.random.rand(1, 10, 15, 1),
            np.random.rand(1, 10, 15, 1),
            np.random.rand(1, 10, 15, 3),
            np.random.rand(1, 10, 15, 3)
        ]
        rand_target = [
            np.random.rand(1, 10, 15, 1),
            np.random.rand(1, 10, 15, 3),
            np.random.rand(1, 10, 15, 1),
            np.random.rand(1, 10, 15, 3),
            np.random.rand(1, 10, 15, 3),
            np.random.rand(1, 10, 15, 1),
            np.random.rand(1, 10, 15, 3),
            np.random.rand(1, 10, 15, 3),
            np.random.rand(1, 10, 15, 1),
            np.random.rand(1, 10, 15, 3)
        ]

        rand_image = np.random.rand(1, 10, 10, 80)
        rand_vectors = np.random.rand(1, 10, 10, 3)

        image = Input(rand_image.shape[1:], name='RandomImageInputForConvolution')
        vectors = Input(rand_vectors.shape[1:], name='RandomVectorsInputForConvolution')
        inps = [
            Input(inpt.shape[1:], name='RandomInput{}ForConvolution'.format(i))
            for i, inpt in enumerate(rand_inputs)
        ]

        # 2 RO0 inputs = 4 outputs, 2 RO1 inputs = 6 output tensors; so 10 total output tensors...
        conv_output = Convolution(image=image, unit_vectors=vectors, filter_dim=15)(inps)
        input_layers = [image, vectors]
        input_layers.extend(inps)
        model = Model(inputs=input_layers, outputs=conv_output)
        inputs = [rand_image, rand_vectors]
        inputs.extend(rand_inputs)
        loss, pred = compile_fit_eval_pred(
            model,
            inputs=inputs,
            targets=rand_target
        )

        assert len(pred) == 10


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
        r = Input(shape=(10, 3), name='Cartesians', dtype='float32')
        z = Input(shape=(10,), name='AtomicNums', dtype='int32')
        one_hot = OneHot(4)(z)
        dist = DistanceMatrix()(r)
        image = GaussianBasis(**{
            'width': 0.2,
            'spacing': 0.2,
            'min_value': -1.0,
            'max_value': 15.0
        })(dist)
        unit_vector = UnitVectors()(
            DifferenceMatrix()(r)
        )

        one_hot_reshape = Lambda(K.expand_dims, name='OneHotReshape')(one_hot)
        embedding = SelfInteraction(name='Embedding', output_dim=15)(one_hot_reshape)
        conv_output = Convolution(image=image, unit_vectors=unit_vector, filter_dim=15, name='Convolution')([embedding])
        cat_output = Concatenation(name='Concatenation')(conv_output)
        si_output = SelfInteraction(name='SelfInteraction1', output_dim=15)(cat_output)
        nonlin_output = Nonlinearity(name='Nonlinearity1')(si_output)
        si_output_2 = SelfInteraction(name='SelfInteraction2', output_dim=1)(nonlin_output)
        predicted_atomic_e = Nonlinearity(name='Nonlinearity2')(si_output_2[0])
        predicted_atomic_e = Reshape((10,))(predicted_atomic_e)

        model = Model(inputs=[r, z], outputs=predicted_atomic_e)
        model.compile(
            optimizer=optimizers.get({
                'class_name': 'adam',
                'config': {
                    'lr': 1e-3
                }
            }),
            loss='mean_squared_error'
        )
        model.fit(
            x=[cart, atom_num],
            y=energy,
            epochs=3
        )
