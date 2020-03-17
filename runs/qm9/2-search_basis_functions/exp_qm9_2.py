from tfn.tools.jobs import Search

job = Search(
    tuner='random',
    exp_config={
        'name': 'QM9 RANDOM SEARCH ON BASIS FUNCTIONS',
        'notes': '9 total models',
        'run_config': {'epochs': 50},
        'loader_config': {'loader_type': 'qm9_loader'},
        'tuner_config': {'max_trials': 9},
        'factory_config': {'factory_type': 'energy_factory'},
        'search_space': {
            'basis_type': {
                'type': 'choice', 'kwargs': {
                    'values': ['gaussian', 'cosine', 'cosine_shifted']
                }
            },
            'basis_width': {
                'type': 'choice', 'kwargs': {
                    'values': [0.1, 0.2, 0.5]
                }
            },
            'basis_spacing': {
                'type': 'choice', 'kwargs': {
                    'values': [0.1, 0.2, 0.5]
                }
            }
        }
    }
)
job.run()
