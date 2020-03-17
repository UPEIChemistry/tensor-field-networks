from tfn.tools.jobs import Search

job = Search(
    tuner='random',
    exp_config={
        'name': 'QM9 RANDOM SEARCH ON RADIALS',
        'notes': '24 total models',
        'run_config': {'epochs': 75},
        'tuner_config': {'max_trials': 24},
        'search_space': {
            'radial_factory': {
                'type': 'choice', 'kwargs': {
                    'values': ['multi_dense', 'single_dense', 'multi_conv', 'single_conv']
                }
            },
            'radial_units': {
                'type': 'choice', 'kwargs': {
                    'values': [32, 64, 128]
                }
            },
            'radial_num_layers': {
                'type': 'choice', 'kwargs': {
                    'values': [2, 3]
                }
            }
        }
    }
)
job.run()
