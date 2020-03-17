from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda, Add

from atomic_images.layers import DistanceMatrix, Unstandardization
from tfn.layers import MolecularSelfInteraction

from . import Builder


class TSBuilder(Builder):
    def get_inputs(self):
        return [
            Input([self.num_atoms, 3], name='reactant_cartesians', dtype='float32'),
            Input([self.num_atoms, 3], name='product_cartesians', dtype='float32'),
            Input([self.num_atoms, ], name='atomic_nums', dtype='int32')
        ]

    def get_learned_output(self, inputs: list):
        r, p, z = inputs
        r_point_cloud = self.point_cloud_layer([r, z])
        p_point_cloud = self.point_cloud_layer([p, z])
        embedding = MolecularSelfInteraction(self.embedding_units, name='embedding')([
                r_point_cloud[0],
                Lambda(lambda x: K.expand_dims(x, axis=-1))(r_point_cloud[0])
            ])  # Expanded one_hot
        layers = self.get_layers()
        reactant_input = self.get_learned_tensors(embedding, r_point_cloud, layers)
        product_input = self.get_learned_tensors(embedding, p_point_cloud, layers)
        return [r_point_cloud, p_point_cloud], [r, p, reactant_input, product_input]

    def get_model_output(self, point_cloud: list, inputs: list):
        r_point_cloud, p_point_cloud = point_cloud
        one_hot = r_point_cloud[0]
        [r, p, reactant_input, product_input] = inputs
        tensors = [Add()([r, p]) for r, p in zip(reactant_input, product_input)]
        tensors = self.get_final_output(one_hot, tensors)
        outputs = []
        if self.use_scalars:
            atomic_energy = Lambda(lambda x: K.squeeze(x, axis=-1), name='atomic_energy')(
                tensors[0])
            atomic_energy = Unstandardization(self.mu, self.sigma, trainable=self.trainable_offsets)(
                [one_hot, atomic_energy]
            )
            molecular_energy = Lambda(lambda x: K.sum(x, axis=-2), name='molecular_energy')(
                atomic_energy)
            outputs.append(molecular_energy)
        vectors = Lambda(lambda x: K.squeeze(x, axis=-2), name='vectors')(tensors[-1])  # mols,
        # atoms, 3
        midpoint = Lambda(
            lambda x: x[0] + ((x[1] - x[0]) / 2),  # M = R + [(R - P) / 2]
            name='midpoint'
        )([r, p])
        cartesians = Add(name='cartesians')([midpoint, vectors])
        outputs.append(DistanceMatrix(name='distance_matrix')(cartesians))
        return outputs
