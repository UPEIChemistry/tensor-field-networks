from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda, Add

from atomic_images.layers import DistanceMatrix, Unstandardization
from tfn.layers import MolecularSelfInteraction

from . import Builder


class TSBuilder(Builder):
    def __init__(self, *args, **kwargs):
        self.output_type = kwargs.pop('output_type', 'dist_matrix')  # [dist_matrix, energy, both]
        super().__init__(*args, **kwargs)

    def get_inputs(self):
        return [
            Input([self.num_atoms, ], name='atomic_nums', dtype='int32'),
            Input([self.num_atoms, 3], name='reactant_cartesians', dtype='float32'),
            Input([self.num_atoms, 3], name='product_cartesians', dtype='float32')
        ]

    def get_learned_output(self, inputs: list):
        z, r, p = inputs
        r_point_cloud = self.point_cloud_layer([r, z])
        p_point_cloud = self.point_cloud_layer([p, z])
        embedding_layer = MolecularSelfInteraction(self.embedding_units, name='embedding')
        reactant_embedding = embedding_layer([
                r_point_cloud[0],
                Lambda(lambda x: K.expand_dims(x, axis=-1))(r_point_cloud[0])
            ])
        product_embedding = embedding_layer([
            p_point_cloud[0],
            Lambda(lambda x: K.expand_dims(x, axis=-1))(p_point_cloud[0])
        ])
        layers = self.get_layers()
        reactant_input = self.get_learned_tensors(reactant_embedding, r_point_cloud, layers)
        product_input = self.get_learned_tensors(product_embedding, p_point_cloud, layers)
        return [r_point_cloud, p_point_cloud], [reactant_input, product_input]

    def get_model_output(self, point_cloud: list, inputs: list):
        r_point_cloud, p_point_cloud = point_cloud
        one_hot = r_point_cloud[0]
        [reactant_input, product_input] = inputs
        tensors = [Add()([r, p]) for r, p in zip(reactant_input, product_input)]
        tensors = self.get_final_output(one_hot, tensors)
        outputs = []
        if self.output_type == 'dist_matrix' or self.output_type == 'both':
            cartesians = Lambda(lambda x: K.squeeze(x, axis=-2), name='vectors')(tensors[-1])
            dist_matrix = DistanceMatrix(name='distance_matrix')(cartesians)
            outputs.append(dist_matrix)
        if self.output_type == 'energy' or self.output_type == 'both':
            atomic_energy = Lambda(lambda x: K.squeeze(x, axis=-1), name='atomic_energy')(
                tensors[0])
            atomic_energy = Unstandardization(self.mu, self.sigma, trainable=self.trainable_offsets)(
                [one_hot, atomic_energy]
            )
            molecular_energy = Lambda(lambda x: K.sum(x, axis=-2), name='molecular_energy')(
                atomic_energy)
            outputs.append(molecular_energy)
        if self.output_type not in ('dist_matrix', 'energy', 'both'):
            raise ValueError(
                'kwarg `output_type` has unknown value: {}'.format(str(self.output_type)))

        return outputs


class TSClassifierBuilder(TSBuilder):
    def get_inputs(self):
        return [
            Input([self.num_atoms, ], name='atomic_nums', dtype='int32'),
            Input([self.num_atoms, 3], name='cartesians', dtype='float32')
        ]


class TSSiameseClassifierBuilder(TSClassifierBuilder):
    def get_inputs(self):
        return [
            Input([2, self.num_atoms, ], name='atomic_nums', dtype='int32'),
            Input([2, self.num_atoms, 3], name='cartesians', dtype='float32')
        ]

    def get_learned_output(self, inputs: list):
        pass
