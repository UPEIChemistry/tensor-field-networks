from tfn.tools.jobs import SingleModel

job = SingleModel(exp_config={
    'name': 'SN2 E/F PREDICTION USING DEFAULT FORCE MODEL',
    'notes': 'Default architectures for both model and radials.',
    'loader_config': {'loader_type': 'sn2_loader'},
    'builder_config': {'builder_type': 'force_builder', 'num_layers': (2, 2, 2),
                       'si_units': (64, 32, 16)}
})
job.run()
