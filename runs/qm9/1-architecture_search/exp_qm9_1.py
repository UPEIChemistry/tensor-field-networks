from tfn.tools.jobs import Search
from tfn.tools.jobs.config_defaults import default_architecture_search

job = Search(
    tuner='random',
    exp_config={
        'name': 'RANDOM ARCHITECTURE SEARCH ON QM9',
        'notes': '500 / 1994 models tested',
        'run_config': {'epochs': 25},
        'loader_config': {'loader_type': 'qm9_loader'},
        'tuner_config': {'max_trials': 500},
        'factory_config': {'factory_type': 'energy_factory'},
        'search_space': default_architecture_search
    }
)
job.run()
