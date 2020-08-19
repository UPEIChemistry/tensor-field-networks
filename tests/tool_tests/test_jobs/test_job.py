from tfn.tools.jobs import KerasJob
from tfn.tools.jobs.config_defaults import loader_config, run_config
from tfn.tools.loaders import DataLoader


class TestJob:
    def test_load_data(self, clear_logdirs):
        job = KerasJob()
        loader, data = job._load_data()
        assert isinstance(loader, DataLoader)
        assert len(data) == 3 and len(data[0]) == 2

    def test_default_config(self, clear_logdirs):
        job = KerasJob()
        assert job.exp_config["loader_config"] == loader_config
        assert job.exp_config["run_config"] == run_config
