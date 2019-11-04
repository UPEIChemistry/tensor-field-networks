import json

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.utils import get_custom_objects
from tfn import layers


class TestRadialFactory:
    def test_get_radial(self):
        _ = layers.DenseRadialFactory().get_radial(32)

    def test_export_and_creation_json(self):
        config = layers.DenseRadialFactory().to_json()
        factory = layers.DenseRadialFactory.from_json(config)
        assert factory.num_layers == 2
        assert factory.units == 32


class TestConvolution:
    def test_defaults(self, random_onehot_rbf_vectors, random_features_and_targets):
        _, *point_cloud = random_onehot_rbf_vectors
        features, targets = random_features_and_targets
        output = layers.Convolution()(list(point_cloud) + list(features))
        assert len(output) == 2

    def test_provided_radial_json(self, random_onehot_rbf_vectors, random_features_and_targets):
        _, *point_cloud = random_onehot_rbf_vectors
        features, targets = random_features_and_targets
        d = {'type': 'DenseRadialFactory', 'num_layers': 3, 'units': 4, 'kernel_lambda': 0.01}
        _ = layers.Convolution(radial_factory=json.dumps(d))(list(point_cloud) + list(features))

    def test_provided_radial_string(self,
                                    random_onehot_rbf_vectors,
                                    random_features_and_targets):
        _, *point_cloud = random_onehot_rbf_vectors
        features, targets = random_features_and_targets
        conv = layers.Convolution(radial_factory='DenseRadialFactory')
        _ = conv(list(point_cloud) + list(features))
        config = json.loads(conv.radial_factory.to_json())
        assert config['type'] == 'DenseRadialFactory'
        assert config['units'] == 32

    def test_provided_radial_string_and_kwargs(self,
                                               random_onehot_rbf_vectors,
                                               random_features_and_targets):
        _, *point_cloud = random_onehot_rbf_vectors
        features, targets = random_features_and_targets
        conv = layers.Convolution(
            radial_factory='DenseRadialFactory',
            factory_kwargs={'num_layers': 3, 'units': 4, 'kernel_lambda': 0.01}
        )
        _ = conv(list(point_cloud) + list(features))
        config = json.loads(conv.radial_factory.to_json())
        assert config['units'] == 4

    def test_custom_radial(self, random_onehot_rbf_vectors, random_features_and_targets):
        class MyRadial(layers.DenseRadialFactory):
            def __init__(self, num_units):
                super().__init__()
                self.num_units = num_units

            def get_radial(self, feature_dim, input_order=None, filter_order=None):
                return Dense(feature_dim)

            @classmethod
            def from_json(cls, json_str: str):
                return cls(**json.loads(json_str))

        get_custom_objects().update({MyRadial.__name__: MyRadial})
        _, *point_cloud = random_onehot_rbf_vectors
        features, targets = random_features_and_targets
        conv = layers.Convolution(radial_factory='MyRadial', factory_kwargs={'num_units': 6})
        _ = conv(list(point_cloud) + list(features))
        config = json.loads(conv.radial_factory.to_json())
        assert config['type'] == 'MyRadial'
        assert config['num_units'] == 6

    def test_get_config(self, random_onehot_rbf_vectors, random_features_and_targets):
        _, *point_cloud = random_onehot_rbf_vectors
        features, targets = random_features_and_targets
        conv = layers.Convolution()
        _ = conv(list(point_cloud) + list(features))
        config = conv.get_config()
        assert config['trainable'] is True and config['si_units'] == 16


class TestMolecularConvolution:
    def test_defaults(self, random_onehot_rbf_vectors, random_features_and_targets):
        features, targets = random_features_and_targets
        output = layers.MolecularConvolution()(list(random_onehot_rbf_vectors) + list(features))
        assert len(output) == 2


class TestHarmonicFilter:
    def test_ro0_filter(self, random_onehot_rbf_vectors):
        _, rbf, vectors = random_onehot_rbf_vectors
        output = layers.HarmonicFilter(
            radial=Dense(16),
            filter_order=0
        )([rbf, vectors])
        assert output.shape[-1] == 1

    def test_ro1_filter(self, random_onehot_rbf_vectors):
        _, rbf, vectors = random_onehot_rbf_vectors
        output = layers.HarmonicFilter(
            radial=Dense(16),
            filter_order=1
        )([rbf, vectors])
        assert output.shape[-1] == 3

    def test_ro2_filter(self, random_onehot_rbf_vectors):
        _, rbf, vectors = random_onehot_rbf_vectors
        output = layers.HarmonicFilter(
            radial=Dense(16),
            filter_order=2
        )([rbf, vectors])
        assert output.shape[-1] == 5


class TestSelfInteraction:
    def test_correct_output_shapes(self, random_features_and_targets):
        inputs, targets = random_features_and_targets
        si = layers.SelfInteraction(32)
        outputs = si(inputs)
        assert outputs[0].shape == (2, 10, 32, 1) and outputs[1].shape == (2, 10, 32, 3)


class TestEquivariantActivation:
    def test_correct_output_shapes(self, random_features_and_targets):
        inputs, targets = random_features_and_targets
        outputs = layers.EquivariantActivation()(inputs)
        assert all([i.shape == o.shape for i, o in zip(inputs, outputs)])


class TestPreprocessing:
    def test_3_output_tensors(self, random_cartesians_and_z):
        r, z = random_cartesians_and_z
        pre_block = layers.Preprocessing(
            max_z=5,
            basis_config={
                'width': 0.2,
                'spacing': 0.2,
                'min_value': -1.0,
                'max_value': 15.0
            }
        )
        outputs = pre_block([r, z])
        assert len(outputs) == 3

    def test_cosine_basis(self, random_cartesians_and_z):
        pre_block = layers.Preprocessing(
            max_z=5,
            basis_type='cosine'
        )
        outputs = pre_block(random_cartesians_and_z)
        assert len(outputs) == 3
        assert outputs[1].shape == (2, 10, 10, 80)
