import json

import numpy as np
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.utils import get_custom_objects
from tfn import layers


class TestRadialFactory:
    def test_get_radial(self):
        _ = layers.DenseRadialFactory().get_radial(32)

    def test_export_and_creation_json(self):
        config = layers.DenseRadialFactory().to_json()
        factory = layers.DenseRadialFactory.from_json(config)
        assert factory.num_layers == 2
        assert factory.units == 32

    def test_radial_sum_atoms(self):
        radial = layers.Radial(units=16, sum_points=True)
        output = radial(np.random.randn(2, 1, 160))
        assert output.shape == (2, 1, 16)


class TestConvolution:
    def test_defaults(self, random_onehot_rbf_vectors, random_features_and_targets):
        _, *point_cloud = random_onehot_rbf_vectors
        features, targets = random_features_and_targets
        output = layers.Convolution()(list(point_cloud) + features)
        assert len(output) == 2

    def test_provided_radial_json(
        self, random_onehot_rbf_vectors, random_features_and_targets
    ):
        _, *point_cloud = random_onehot_rbf_vectors
        features, targets = random_features_and_targets
        d = {
            "type": "DenseRadialFactory",
            "num_layers": 3,
            "units": 4,
            "kernel_lambda": 0.01,
        }
        _ = layers.Convolution(radial_factory=json.dumps(d))(
            list(point_cloud) + list(features)
        )

    def test_provided_radial_string(
        self, random_onehot_rbf_vectors, random_features_and_targets
    ):
        _, *point_cloud = random_onehot_rbf_vectors
        features, targets = random_features_and_targets
        conv = layers.Convolution(radial_factory="DenseRadialFactory")
        _ = conv(list(point_cloud) + list(features))
        config = json.loads(conv.radial_factory.to_json())
        assert config["type"] == "DenseRadialFactory"
        assert config["units"] == 32

    def test_provided_radial_string_and_kwargs(
        self, random_onehot_rbf_vectors, random_features_and_targets
    ):
        _, *point_cloud = random_onehot_rbf_vectors
        features, targets = random_features_and_targets
        conv = layers.Convolution(
            radial_factory="DenseRadialFactory",
            factory_kwargs={"num_layers": 3, "units": 4, "kernel_lambda": 0.01},
        )
        _ = conv(list(point_cloud) + list(features))
        config = json.loads(conv.radial_factory.to_json())
        assert config["units"] == 4

    def test_custom_radial(
        self, random_onehot_rbf_vectors, random_features_and_targets
    ):
        class MyRadial(layers.DenseRadialFactory):
            def __init__(self, num_units, **kwargs):
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
        conv = layers.Convolution(
            radial_factory="MyRadial", factory_kwargs={"num_units": 6}
        )
        _ = conv(list(point_cloud) + list(features))
        config = json.loads(conv.radial_factory.to_json())
        assert config["type"] == "MyRadial"
        assert config["num_units"] == 6

    def test_get_config(self, random_onehot_rbf_vectors, random_features_and_targets):
        _, *point_cloud = random_onehot_rbf_vectors
        features, targets = random_features_and_targets
        conv = layers.Convolution()
        _ = conv(list(point_cloud) + list(features))
        config = conv.get_config()
        assert config["trainable"] is True and config["si_units"] == 16


class TestMolecularConvolution:
    def test_defaults(self, random_onehot_rbf_vectors, random_features_and_targets):
        features, targets = random_features_and_targets
        output = layers.MolecularConvolution()(
            list(random_onehot_rbf_vectors) + list(features)
        )
        assert len(output) == 2

    def test_sum_points(self, random_z_and_cartesians, random_features_and_targets):
        point_cloud = layers.Preprocessing(max_z=5, sum_points=True)(
            random_z_and_cartesians
        )
        assert len(point_cloud[1].shape) == 3
        features, targets = random_features_and_targets
        output = layers.MolecularConvolution(sum_points=True)(point_cloud + features)
        assert len(output) == 2

    def test_one_in_one_out(self, random_onehot_rbf_vectors):
        point_cloud = random_onehot_rbf_vectors
        features = np.random.rand(2, 1, 1, 1).astype("float32")
        output = layers.MolecularConvolution(si_units=1)(list(point_cloud) + [features])
        assert len(output) == 2


class TestHarmonicFilter:
    def test_ro0_filter(self, random_onehot_rbf_vectors):
        _, rbf, vectors = random_onehot_rbf_vectors
        output = layers.HarmonicFilter(radial=Dense(16), filter_order=0)([rbf, vectors])
        assert output.shape[-1] == 1

    def test_ro1_filter(self, random_onehot_rbf_vectors):
        _, rbf, vectors = random_onehot_rbf_vectors
        output = layers.HarmonicFilter(radial=Dense(16), filter_order=1)([rbf, vectors])
        assert output.shape[-1] == 3

    def test_ro2_filter(self, random_onehot_rbf_vectors):
        _, rbf, vectors = random_onehot_rbf_vectors
        output = layers.HarmonicFilter(radial=Dense(16), filter_order=2)([rbf, vectors])
        assert output.shape[-1] == 5


class TestSelfInteraction:
    def test_correct_output_shapes(self, random_features_and_targets):
        inputs, targets = random_features_and_targets
        si = layers.SelfInteraction(32)
        outputs = si(inputs)
        assert outputs[0].shape == (2, 1, 32, 1) and outputs[1].shape == (2, 1, 32, 3)

    def test_molecular_si(self, random_features_and_targets, random_onehot_rbf_vectors):
        one_hot, *_ = random_onehot_rbf_vectors
        inputs, targets = random_features_and_targets
        si = layers.MolecularSelfInteraction(32)
        outputs = si([one_hot] + inputs)
        assert outputs[0].shape == (2, 1, 32, 1) and outputs[1].shape == (2, 1, 32, 3)

    def test_one_to_one_si(self):
        inputs = [np.random.rand(2, 10, 1, 1).astype("float32")]
        output = layers.SelfInteraction(1)(inputs)[0]
        assert output.shape == (2, 10, 1, 1)


class TestEquivariantActivation:
    def test_correct_output_shapes(self, random_features_and_targets):
        inputs, targets = random_features_and_targets
        outputs = layers.EquivariantActivation()(inputs)
        assert all([i.shape == o.shape for i, o in zip(inputs, outputs)])

    def test_molecular_activation(
        self, random_features_and_targets, random_onehot_rbf_vectors
    ):
        one_hot, *_ = random_onehot_rbf_vectors
        inputs, targets = random_features_and_targets
        outputs = layers.MolecularActivation()([one_hot] + inputs)
        assert all([i.shape == o.shape for i, o in zip(inputs, outputs)])


class TestPreprocessing:
    def test_3_output_tensors(self, random_z_and_cartesians):
        pre_block = layers.Preprocessing(
            max_z=5,
            basis_config={
                "width": 0.2,
                "spacing": 0.2,
                "min_value": -1.0,
                "max_value": 15.0,
            },
            dynamic=True,
        )
        outputs = pre_block(random_z_and_cartesians)
        assert len(outputs) == 3

    def test_cosine_basis(self, random_z_and_cartesians):
        pre_block = layers.Preprocessing(max_z=5, basis_type="cosine")
        outputs = pre_block(random_z_and_cartesians)
        assert len(outputs) == 3
        assert pre_block.basis_type == "cosine"
        assert outputs[1].shape == (2, 1, 1, 80)

    def test_shifted_cosine_basis(self, random_z_and_cartesians):
        pre_block = layers.Preprocessing(max_z=5, basis_type="shifted_cosine")
        outputs = pre_block(random_z_and_cartesians)
        assert len(outputs) == 3
        assert pre_block.basis_type == "shifted_cosine"
        assert outputs[1].shape == (2, 1, 1, 80)
