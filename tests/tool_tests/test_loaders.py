import pytest
import os

from math import ceil, isclose
import numpy as np

from tfn.tools.loaders import (
    ISO17DataLoader,
    QM9DataDataLoader,
    TSLoader,
    SN2Loader,
    IsomLoader,
)


class TestQM9Loader:
    def test_load_data(self):
        loader = QM9DataDataLoader(os.environ["DATADIR"] + "/QM9_data_original.hdf5")
        data = loader.load_data()
        assert len(data) == 3
        assert len(data[0]) == 2
        assert len(data[0][0]) == 2

    def test_train_val_test_splitting(self):
        loader = QM9DataDataLoader(
            os.environ["DATADIR"] + "/QM9_data_original.hdf5", splitting="70:20:10"
        )
        data = loader.load_data()
        assert len(data) == 3
        assert isclose(len(data[0][0][0]), ceil(0.70 * 133885), abs_tol=1)
        assert isclose(len(data[1][0][0]), ceil(0.20 * 133885), abs_tol=1)
        assert isclose(len(data[2][0][0]), ceil(0.10 * 133885), abs_tol=1)

    def test_train_val_splitting(self):
        loader = QM9DataDataLoader(
            os.environ["DATADIR"] + "/QM9_data_original.hdf5", splitting="70:30:0"
        )
        data = loader.load_data()
        assert len(data) == 3
        assert isclose(len(data[0][0][0]), ceil(0.70 * 133885), abs_tol=1)
        assert isclose(len(data[1][0][0]), ceil(0.30 * 133885), abs_tol=1)
        assert data[2] is None

    def test_train_test_splitting(self):
        loader = QM9DataDataLoader(
            os.environ["DATADIR"] + "/QM9_data_original.hdf5", splitting="70:0:30"
        )
        data = loader.load_data()
        assert len(data) == 3
        assert isclose(len(data[0][0][0]), ceil(0.70 * 133885), abs_tol=1)
        assert data[1] is None
        assert isclose(len(data[2][0][0]), ceil(0.30 * 133885), abs_tol=1)

    def test_cross_validation_splitting(self):
        loader = QM9DataDataLoader(
            os.environ["DATADIR"] + "/QM9_data_original.hdf5", splitting=7
        )
        data = loader.load_data()
        assert len(data) == 7
        assert len(np.concatenate([d[0][0] for d in data], axis=0)) == 133885

    def test_modified(self):
        loader = QM9DataDataLoader(
            os.environ["DATADIR"] + "/QM9_data_original.hdf5", splitting="70:20:10"
        )
        data = loader.load_data(modify_structures=True, modify_distance=1)
        assert len(data[0][0]) == 3
        assert data[0][1][0].shape[1:] == (loader.num_points, 3)

    def test_distance_matrix(self):
        loader = QM9DataDataLoader(
            os.environ["DATADIR"] + "/QM9_data_original.hdf5", splitting="70:20:10"
        )
        data = loader.load_data(
            modify_structures=True, modify_distance=1, output_distance_matrix=True
        )
        assert len(data[0][0]) == 3
        assert data[0][1][0].shape[1:] == (loader.num_points, loader.num_points)

    def test_classifier(self):
        loader = QM9DataDataLoader(
            os.environ["DATADIR"] + "/QM9_data_original.hdf5", splitting="70:20:10"
        )
        data = loader.load_data(modify_structures=True, classifier_output=True)
        assert len(data[0][0]) == 2  # tiled atomic_nums, tiled cartesians
        assert len(data[0][1][0].shape) == 1


class TestISO17Loader:
    def test_load_dual_data(self):
        loader = ISO17DataLoader(os.environ["DATADIR"] + "/iso17.hdf5")
        data = loader.load_data()
        assert len(data) == 3  # train, val, test
        assert len(data[0]) == 2  # x_train, y_train
        assert len(data[0][0]) == 2  # r, z
        assert (
            ceil(len(data[0][0][0]) / 0.70) == 461715
        )  # ensure train split is 95% of total reference examples
        assert data[0][0][0].shape[1] == 29

    def test_load_force_data(self):
        loader = ISO17DataLoader(
            os.environ["DATADIR"] + "/iso17.hdf5", use_energies=False
        )
        data = loader.load_data()
        assert len(data[0][1]) == 1  # Only 1 y values


class TestTSLoader:
    def test_load_ts_data(self):
        loader = TSLoader(os.environ["DATADIR"] + "/ts.hdf5")
        data = loader.load_data()
        assert len(data) == 3
        assert len(data[0]) == 2
        assert len(data[0][0]) == 3  # Z, R, P
        assert len(data[0][1]) == 1  # TS

    def test_complexes(self):
        loader = TSLoader(os.environ["DATADIR"] + "/ts.hdf5")
        data = loader.load_data("use_complexes")
        assert len(data) == 3
        assert len(data[0]) == 2
        assert len(data[0][0]) == 3  # Z, RC, PC
        assert len(data[0][1]) == 1  # TS

    def test_remove_noise(self):
        loader = TSLoader(os.environ["DATADIR"] + "/ts.hdf5", splitting=None)
        data = loader.load_data(remove_noise=True)
        assert len(data[0][0][0]) == 55

    def test_energy_serving(self):
        loader = TSLoader(os.environ["DATADIR"] + "/ts.hdf5", pre_load=False)
        data = loader.load_data(output_type="energies")
        assert len(data) == 3
        assert len(data[0][0]) == 3
        assert len(data[0][1][0].shape) == 1

    def test_serving_both(self):
        loader = TSLoader(os.environ["DATADIR"] + "/ts.hdf5", pre_load=False)
        data = loader.load_data(output_type="both")
        assert len(data) == 3
        assert len(data[0][0]) == 3
        assert len(data[0][1]) == 2
        assert data[0][1][0].shape[1:] == (loader.num_points, 3)
        assert len(data[0][1][1].shape) == 1

    def test_distance_matrix(self):
        loader = TSLoader(os.environ["DATADIR"] + "/ts.hdf5", pre_load=False)
        data = loader.load_data(output_type="both", output_distance_matrix=True)
        assert len(data) == 3
        assert len(data[0][0]) == 3
        assert len(data[0][1]) == 2
        assert data[0][1][0].shape[1:] == (loader.num_points, loader.num_points)
        assert len(data[0][1][1].shape) == 1

    def test_classification_data(self):
        loader = TSLoader(os.environ["DATADIR"] + "/ts.hdf5", pre_load=False)
        data = loader.load_data(output_type="classifier")
        assert len(data) == 3
        assert len(data[0][0]) == 2
        assert len(data[0][1]) == 1
        assert len(data[0][1][0].shape) == 1

    def test_siamese_data(self):
        loader = TSLoader(os.environ["DATADIR"] + "/ts.hdf5", pre_load=False)
        data = loader.load_data(output_type="siamese")
        assert len(data) == 3
        assert data[0][0][0].shape[1:] == (2, loader.num_points,)
        assert data[0][0][1].shape[1:] == (2, loader.num_points, 3)
        assert len(data[0][1]) == 1

    def test_custom_split(self):
        loader = TSLoader(
            os.environ["DATADIR"] + "/ts.hdf5", pre_load=False, splitting="custom"
        )
        train, val, test = loader.load_data(shuffle=False)
        assert test is None
        assert len(val[0][0]) == 7


class TestSN2Loader:
    def test_load_sn2_data(self):
        loader = SN2Loader(os.environ["DATADIR"] + "/sn2_reactions.npz")
        data = loader.load_data()
        assert len(data) == 3
        assert len(data[0]) == 2
        assert len(data[0][0]) == 2  # R, Z
        assert len(data[0][1]) == 2  # E, F
        assert len(loader.mu) == 2
        assert isclose(loader.sigma[1], 0.71, abs_tol=0.3)


class TestIsomLoader:
    def test_load_isomerization_data(self):
        loader = IsomLoader(
            "/home/riley/Documents/tensor-field-networks/data/isomerization/isomerization_dataset.hd5f"
        )
        data = loader.load_data()
        assert data
