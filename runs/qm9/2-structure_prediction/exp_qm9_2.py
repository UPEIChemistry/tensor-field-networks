from tfn.tools.jobs import SingleModel

job = SingleModel(exp_config={
    'name': 'QM9 E PREDICTION USING DEFAULT ENERGY MODEL',
    'notes': 'Using all defaults',
    'loader_config': {'loader_type': 'qm9_loader', 'load_kwargs': {'modify_structures': True}},
    'builder_config': {'builder_type': 'ts_builder', 'num_layers': (2, 2, 2),
                       'si_units': (64, 32, 16)}
})
job.run()
