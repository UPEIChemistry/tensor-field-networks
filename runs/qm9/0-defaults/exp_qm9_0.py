from tfn.tools.jobs import SingleModel

job = SingleModel(exp_config={
    'name': 'QM9 E PREDICTION USING DEFAULT ENERGY MODEL',
    'notes': 'Using all defaults',
    'builder_config': {'num_layers': (4, 3, 2), 'si_units': (32, 16, 8)}
})
job.run()
