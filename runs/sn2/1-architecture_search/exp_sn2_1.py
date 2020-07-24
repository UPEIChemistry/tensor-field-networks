from tfn.tools.jobs import GridSearch
from tfn.tools.jobs.config_defaults import default_grid_search

job = GridSearch(
    total_models=100,
    exp_config={
        'name': 'SN2 architecture grid search',
        'notes': 'Testing all 96 models of default grid search',
        'run_config': {'epochs': 20, 'loss_weights': [0.01, 1], 'save_model': False},
        'loader_config': {'loader_type': 'sn2_loader'},
        'builder_config': {'builder_type': 'force_builder'},
        'grid_config': default_grid_search
    }
)
job.run()
