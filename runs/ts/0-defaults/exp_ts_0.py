from tfn.tools.jobs import SingleModel

job = SingleModel(exp_config={
    'name': 'DEFAULT TS MODEL ON TS DATASET',
    'notes': 'Train on only distance matrix',
    'run_config': {'epochs': 200, 'loss': 'mae', 'optimizer_kwargs': {
        'learning_rate': 0.001}},
    'builder_config': {'builder_type': 'ts_builder', 'num_layers': (2, 2, 2),
                       'si_units': 64},
    'loader_config': {
        'loader_type': 'ts_loader',
        'load_kwargs': {
            'output_distance_matrix': True,
            'use_complexes': False
        }
    }
})
job.run()
