from tensorflow.keras.models import load_model

from tfn.tools.jobs import SingleModel


class TestScalarModels:
    def test_defaults(self, run_config, builder_config):
        job = SingleModel({
            'name': 'test',
            'run_config': run_config,
            'builder_config': builder_config
        })
        job.run()

    def test_cosine_basis(self, run_config, builder_config):
        job = SingleModel({
            'name': 'test',
            'run_config': run_config,
            'builder_config': dict(**builder_config, basis_type='cosine')
        })
        job.run()

    def test_multi_conv_radial(self, run_config, builder_config):
        job = SingleModel({
            'name': 'test',
            'run_config': run_config,
            'builder_config': dict(**builder_config, radial_factory='multi_conv')
        })
        job.run()

    def test_single_conv_radial(self, run_config, builder_config):
        job = SingleModel({
            'name': 'test',
            'run_config': run_config,
            'builder_config': dict(**builder_config, radial_factory='single_conv')
        })
        job.run()

    def test_default_loads_graphly(self, run_config, builder_config, model):
        run_config['run_eagerly'] = False
        builder_config['dynamic'] = False
        job = SingleModel({
            'name': 'test',
            'run_config': run_config,
            'builder_config': builder_config
        })
        job.run()
        model = load_model(model)
        assert True

    def test_default_loads_eagerly(self, run_config, builder_config, model):
        run_config['run_eagerly'] = True
        builder_config['dynamic'] = True
        job = SingleModel({
            'name': 'test',
            'run_config': run_config,
            'builder_config': builder_config
        })
        job.run()
        model = load_model(model)
        assert True


class TestDualModels:
    def test_defaults(self, run_config, builder_config):
        loader_config = {'loader_type': 'iso17_loader'}
        job = SingleModel({
            'name': 'test',
            'run_config': run_config,
            'loader_config': loader_config,
            'builder_config': dict(**builder_config, builder_type='force_builder')
        })
        job.run()

    def test_sn2(self, run_config, builder_config):
        loader_config = {'loader_type': 'sn2_loader'}
        job = SingleModel({
            'name': 'test',
            'run_config': run_config,
            'loader_config': loader_config,
            'builder_config': dict(**builder_config, builder_type='force_builder')
        })
        job.run()


class TestVectorModels:
    def test_ts_job(self, run_config, builder_config):
        loader_config = {'loader_type': 'ts_loader'}
        job = SingleModel({
            'name': 'test',
            'run_config': run_config,
            'loader_config': loader_config,
            'builder_config': dict(**builder_config, builder_type='ts_builder')
        })
        job.run()
