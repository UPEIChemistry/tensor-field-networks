from datetime import datetime


# General to all Jobs
run_config = {  # Defines kwargs used when running a job
    'storage_dir': './sacred_storage',
    'model_path': './model.hdf5',
    'epochs': 100,
    'test': True,
    'use_strategy': True,
    'loss': 'mae',
    'optimizer': 'adam',
    'loss_weights': None,
    'class_weight': None,
    'metrics': None,
    'run_eagerly': False,
    'select_few': False,
    'capture_output': True,
    'num_models_to_test': 1,
    'fit_verbosity': 2
}
loader_config = {  # Passed directly to loader classes
    'loader_type': 'qm9_loader',
    'map_atoms': True,
    'splitting': '85:10:5',
    'pre_load': False,
    'load_kwargs': {
        'cache': True
    }
}

# SingleModel Only
builder_config = {  # Passed directly to builder classes
    'builder_type': 'energy_builder',
    'standardize': True,
    'trainable_offsets': True,
    'name': 'model',
    'embedding_units': 32,
    'num_layers': 3,
    'si_units': 32,
    'residual': True,
    'activation': 'ssp',
    'dynamic': False,
    'basis_type': 'gaussian',
    'basis_config': {
        'width': 0.2,
        'spacing': 0.2,
        'min_value': -1.0,
        'max_value': 15.0
    },
    'num_final_si_layers': 0,
    'final_si_units': 32,
    'radial_factory': 'multi_dense',
    'radial_kwargs': {
        'num_layers': 2,
        'units': 32,
        'activation': 'ssp',
        'kernel_lambda': 0.01,
        'bias_lambda': 0.01
    }
}

# Search Only
factory_config = dict(  # Passed directly to factory class
    **builder_config,
    **run_config,
    **{
        'factory_type': 'energy_factory'
    }
)

# Pipeline only
pipeline_config = {
    'configs': [
        {'builder_config': builder_config, 'loader_config': loader_config},
        {'builder_config': builder_config, 'loader_config': loader_config}
    ],
    'freeze_layers': False,
}

tuner_config = {  # Passed directly to tuner classes
    'project_name': datetime.utcnow().strftime("%Y-%m-%d-%H%MZ"),
    'directory': './tuner_storage',
    'objective': 'val_loss'
}

# Callbacks
tb_config = {  # Passed directly to tensorboard callback
    'log_dir': './sacred_storage/logs',
    'histogram_freq': 0,
    'profile_batch': 0
}
lr_config = {  # Passed directly to ReduceLROnPlateau callback
    'monitor': 'val_loss',
    'factor': 0.1,
    'patience': 8,
    'verbose': 1,
    'min_delta': 0.001,
    'cooldown': 5,
    'min_lr': 0.00001
}


# Precanned search-spaces
default_architecture_search = {  # 1994 possible models
    'si_units': {
        'type': 'choice', 'kwargs': {
            'values': [16, 32, 64, 128]
        }
    },
    'model_num_layers': {
        'type': 'choice', 'kwargs': {
            'values': [4, 8, 16, 32, 64, 128]
        }
    },
    'num_final_si_layers': {
        'type': 'choice', 'kwargs': {
            'values': [0, 1, 2]
        }
    },
    'final_si_units': {
        'type': 'choice', 'kwargs': {
            'values': [8, 16, 32]
        }
    },
    'radial_num_layers': {
        'type': 'choice', 'kwargs': {
            'values': [1, 2, 3]
        }
    },
    'radial_units': {
        'type': 'choice', 'kwargs': {
            'values': [16, 32, 64]
        }
    }
}
