import os

from math import ceil, isclose

from tfn.tools.loaders import ISO17DataLoader, QM9DataDataLoader, TSLoader, SN2Loader


class TestQM9Loader:
    def test_load_data(self):
        loader = QM9DataDataLoader(os.environ['DATADIR'] + '/QM9_data_original.hdf5')
        data = loader.load_data()
        assert len(data) == 3
        assert len(data[0]) == 2
        assert len(data[0][0]) == 2
        assert ceil(len(data[0][0][0]) / .7) == 133885.

    def test_splitting(self):
        loader = QM9DataDataLoader(os.environ['DATADIR'] + '/QM9_data_original.hdf5',
                                   splitting='70:20:10')
        data = loader.load_data()
        assert len(data) == 3
        assert isclose(len(data[0][0][0]), ceil(0.70 * 133885), abs_tol=1)
        assert isclose(len(data[1][0][0]), ceil(0.20 * 133885), abs_tol=1)
        assert isclose(len(data[2][0][0]), ceil(0.10 * 133885), abs_tol=1)


class TestISO17Loader:
    def test_load_dual_data(self):
        loader = ISO17DataLoader(os.environ['DATADIR'] + '/iso17.hdf5')
        data = loader.load_data()
        assert len(data) == 3  # train, val, test
        assert len(data[0]) == 2  # x_train, y_train
        assert len(data[0][0]) == 2  # r, z
        assert ceil(len(data[0][0][0]) / .70) == 404000.  # ensure train split is 95% of total reference examples
        assert data[0][0][0].shape[1] == 29

    def test_load_force_data(self):
        loader = ISO17DataLoader(os.environ['DATADIR'] + '/iso17.hdf5', use_energies=False)
        data = loader.load_data()
        assert len(data[0][1]) == 1  # Only 1 y values


class TestTSLoader:
    def test_load_ts_data(self):
        loader = TSLoader(os.environ['DATADIR'] + '/ts.hdf5')
        data = loader.load_data()
        assert len(data) == 3
        assert len(data[0]) == 2
        assert len(data[0][0]) == 3  # Z, R, P
        assert len(data[0][1]) == 1  # TS

    def test_complexes(self):
        loader = TSLoader(os.environ['DATADIR'] + '/ts.hdf5')
        data = loader.load_data('use_complexes')
        assert len(data) == 3
        assert len(data[0]) == 2
        assert len(data[0][0]) == 3  # Z, RC, PC
        assert len(data[0][1]) == 1  # TS

    def test_energy_serving(self):
        loader = TSLoader(os.environ['DATADIR'] + '/ts.hdf5', pre_load=False)
        data = loader.load_data(output_type='energies')
        assert len(data) == 3
        assert len(data[0][0]) == 5
        assert len(data[0][1][0].shape) == 1

    def test_serving_both(self):
        loader = TSLoader(os.environ['DATADIR'] + '/ts.hdf5', pre_load=False)
        data = loader.load_data(output_type='both')
        assert len(data) == 3
        assert len(data[0][0]) == 5
        assert len(data[0][1]) == 2
        assert data[0][1][0].shape[1:] == (loader.num_atoms, 3)
        assert len(data[0][1][1].shape) == 1

    def test_distance_matrix(self):
        loader = TSLoader(os.environ['DATADIR'] + '/ts.hdf5', pre_load=False)
        data = loader.load_data(output_type='both', output_distance_matrix=True)
        assert len(data) == 3
        assert len(data[0][0]) == 5
        assert len(data[0][1]) == 2
        assert data[0][1][0].shape[1:] == (loader.num_atoms, loader.num_atoms)
        assert len(data[0][1][1].shape) == 1

    def test_classification_data(self):
        loader = TSLoader(os.environ['DATADIR'] + '/ts.hdf5', pre_load=False)
        data = loader.load_data(output_type='classifier')
        assert len(data) == 3
        assert len(data[0][0]) == 2
        assert len(data[0][1]) == 1
        assert len(data[0][1][0].shape) == 1

    def test_siamese_data(self):
        loader = TSLoader(os.environ['DATADIR'] + '/ts.hdf5', pre_load=False)
        data = loader.load_data(output_type='siamese')
        assert len(data) == 3
        assert data[0][0][0].shape[1:] == (2, loader.num_atoms,)
        assert data[0][0][1].shape[1:] == (2, loader.num_atoms, 3)
        assert len(data[0][1]) == 1


class TestSN2Loader:
    def test_load_sn2_data(self):
        loader = SN2Loader(os.environ['DATADIR'] + '/sn2_reactions.npz')
        data = loader.load_data()
        assert len(data) == 3
        assert len(data[0]) == 2
        assert len(data[0][0]) == 2  # R, Z
        assert len(data[0][1]) == 2  # E, F
        assert len(loader.mu) == 2
        assert isclose(loader.sigma[1], 0.71, abs_tol=0.01)
