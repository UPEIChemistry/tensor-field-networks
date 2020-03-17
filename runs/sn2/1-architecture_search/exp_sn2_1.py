from tfn.tools.jobs import Search
from tfn.tools.jobs.config_defaults import default_architecture_search

job = Search(
    tuner='random',
    exp_config={
        'name': 'RANDOM ARCHITECTURE SEARCH ON SN2',
        'notes': '250 / 1994 models tested',
        'run_config': {'epochs': 25},
        'loader_config': {'loader_type': 'sn2_loader'},
        'tuner_config': {'max_trials': 250},
        'factory_config': {'factory_type': 'force_factory'},
        'search_space': default_architecture_search
    }
)
job.run()
