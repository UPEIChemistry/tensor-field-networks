from tfn.tools.jobs import Search

job = Search(
    tuner='random',
    exp_config={
        'name': 'SN2 RANDOM SEARCH ON LOSS WEIGHTINGS',
        'notes': '25 total models',
        'run_config': {'epochs': 75},
        'loader_config': {'loader_type': 'sn2_loader'},
        'tuner_config': {'max_trials': 25},
        'factory_config': {'factory_type': 'force_factory'},
        'search_space': {
            'molecular_energy_weight': {
                'type': 'choice', 'kwargs': {
                    'values': [0.01, 0.1, 0.25, 0.5, 1.]
                }
            },
            'atomic_forces_weight': {
                'type': 'choice', 'kwargs': {
                    'values': [0.01, 0.1, 0.25, 0.5, 1.]
                }
            }
        }
    }
)
job.run()
