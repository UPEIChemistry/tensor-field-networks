from tfn.tools.jobs import Pipeline
from tfn.tools.jobs.config_defaults import loader_config


class TestPipeline:
    def test_energy_to_energy(self, builder_config, run_config):
        builder_config_0 = dict(**builder_config, name='model_0')
        builder_config_1 = dict(**builder_config, name='model_1')
        pipeline_config = {
            'configs': [
                {'loader_config': loader_config, 'builder_config': builder_config_0},
                {'loader_config': loader_config, 'builder_config': builder_config_1}
            ]
        }
        job = Pipeline({
            'name': 'test',
            'run_config': run_config,
            'pipeline_config': pipeline_config
        })
        job.run()

    def test_energy_to_force(self, builder_config, run_config):
        force_builder_config = dict(**builder_config, builder_type='force_builder',
                                    name='ISO17_energy_and_force_model')
        force_loader_config = {'loader_type': 'iso17_loader'}
        pipeline_config = {
            'configs': [
                {'loader_config': loader_config, 'builder_config': builder_config},
                {'loader_config': force_loader_config, 'builder_config': force_builder_config}
            ]
        }
        job = Pipeline({
            'name': 'test',
            'run_config': run_config,
            'pipeline_config': pipeline_config
        })
        job.run()
