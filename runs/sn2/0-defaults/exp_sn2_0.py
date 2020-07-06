from tfn.tools.jobs import SingleModel

job = SingleModel(exp_config={
    'name': 'SN2 E/F PREDICTION USING DEFAULT FORCE MODEL',
    'notes': 'Default architectures for both model and radials.',
    'loader_config': {'loader_type': 'sn2_loader'},
    'builder_config': {'builder_type': 'force_builder'}
})
job.run()
