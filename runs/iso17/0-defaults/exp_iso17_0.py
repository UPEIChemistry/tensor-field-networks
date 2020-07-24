from tfn.tools.jobs import SingleModel

job = SingleModel(exp_config={
    'name': 'ISO17 E/F PREDICTION USING DEFAULT FORCE MODEL',
    'notes': 'Default architectures for both model and radials.',
    'run_config': {'loss_weights': [0.01, 1], 'fit_verbosity': 1},
    'loader_config': {'loader_type': 'iso17_loader'},
    'builder_config': {'builder_type': 'force_builder',
                       'num_layers': (2, 2, 2),
                       'si_units': (32, 32, 16)}
})
job.run()
