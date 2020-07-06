from tfn.tools.jobs import SingleModel
import os

os.environ['DATADIR'] = '/home/riley/dev/python/data'

job = SingleModel(exp_config={
    'name': 'SIAMESE CLASSIFIER ON TS DATASET EXPERIMENT 0',
    'notes': 'Using all defaults',
    'run_config': {
        'fit_verbosity': 1,
        'class_weights': {0: 1, 1: 5},
        'loss': 'binary_crossentropy',
        'metrics': ['accuracy']
    },
    'loader_config': {
        'loader_type': 'ts_loader',
        'load_kwargs': {'output_type': 'siamese'}
    },
    'builder_config': {
        'builder_type': 'siamese_builder',
        'num_layers': (4, 4, 4),
        'si_units': (32, 32, 16),
        'basis_config': {
            'width': 0.2,
            'spacing': 0.05,
            'min_value': -1.0,
            'max_value': 10.0
        }
    }
})
job.run()
