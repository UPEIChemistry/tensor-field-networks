import json
from tensorflow.python.keras.layers import Dense
from tfn import layers


class TestRadialFactory:
    def test_get_radial(self):
        _ = layers.RadialFactory().get_radial(32)

    def test_export_and_creation_json(self):
        config = json.loads(layers.RadialFactory().to_json())
        factory = layers.RadialFactory.from_json(config)
        assert factory.num_layers == 2
        assert factory.units == 16


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
            gaussian_config={
                'width': 0.2,
                'spacing': 0.2,
                'min_value': -1.0,
                'max_value': 15.0
            }
        )
        outputs = pre_block([r, z])
        assert len(outputs) == 3
