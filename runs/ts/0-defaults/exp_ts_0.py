from tfn.tools.jobs import SingleModel

job = SingleModel(exp_config={
    'name': 'DEFAULT TS MODEL ON TS DATASET',
    'notes': 'Train on only distance matrix',
    'builder_config': {'builder_type': 'ts_builder', 'use_scalars': False},
    'loader_config': {'loader_type': 'ts_loader', 'splitting': '90:10'}
})
job.run()
