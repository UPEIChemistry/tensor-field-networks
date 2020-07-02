from tfn.tools.jobs import SingleModel

job = SingleModel(exp_config={
    'name': 'SIAMESE CLASSIFIER ON TS DATASET EXPERIMENT 0',
    'notes': 'Using all defaults',
    'run_config': {
        'fit_verbosity': 1
    },
    'loader_config': {
        'loader_type': 'ts_loader',
        'load_kwargs': {'output_type': 'siamese'}
    },
    'builder_config': {
        'builder_type': 'siamese_builder',
        'num_layers': (4, 3, 2),
        'si_units': (32, 16, 8)
    }
})
job.run()
