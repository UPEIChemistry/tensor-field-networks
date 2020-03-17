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
        assert len(data[0][0]) == 3  # R, P, Z
        assert len(data[0][1]) == 2  # TS


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
