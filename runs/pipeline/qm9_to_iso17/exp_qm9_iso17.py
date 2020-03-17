from tfn.tools.jobs import Pipeline


job = Pipeline({
    'run_config': {
        'name': 'tl0 - QM9 Energy to ISO17 Energy/Force predictor',
        'notes': 'Not freezing layers, defaults for both models'
    },
    'pipeline_config': {
        'configs': [
            {  # Energy config
                'builder_config': {
                    'builder_type': 'energy_builder'
                },
                'loader_config': {
                    'loader_type': 'qm9_loader'
                }
            },
            {  # Force config
                'builder_config': {
                    'builder_type': 'force_builder'
                },
                'loader_config': {
                    'loader_type': 'iso17_loader',
                    'splitting': '10:2:88'
                }
            }
        ]
    }
})
job.run()
