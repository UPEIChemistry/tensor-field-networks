from datetime import datetime


# General to all Jobs
run_config = {  # Defines kwargs used when running a job
    'storage_dir': './sacred_storage',
    'model_path': './model.hdf5',
    'epochs': 100,
    'batch_size': 32,
    'test': True,
    'save_model': True,
    'use_strategy': False,
    'loss': 'mae',
    'optimizer_kwargs': {'learning_rate': 0.001},
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
    'embedding_units': 64,
    'num_layers': 3,
    'si_units': 64,
    'max_filter_order': 1,
    'residual': True,
    'activation': 'ssp',
    'dynamic': False,
    'sum_atoms': False,
    'basis_type': 'gaussian',
    'basis_config': {  # 321 functions
        'width': 0.2,
        'spacing': 0.05,
        'min_value': -1.0,
        'max_value': 15.0
    },
    'num_final_si_layers': 1,
    'final_si_units': 32,
    'radial_factory': 'multi_dense',
    'radial_kwargs': {
        'num_layers': 2,
        'units': 64,
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
    'factor': 0.5,
    'patience': 5,
    'verbose': 1,
    'min_delta': 0.0001,
    'cooldown': 10,
    'min_lr': 0.000001
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


default_grid_search = {  # 96 models
    'sum_atoms': [True, False],  # 2
    'model_num_layers': [  # 6
        [2 for _ in range(i + 1)]
        for i in [0, 2, 4, 8, 16, 32]  # largest number of clusters (layers = 2 * clusters)
    ],
    'radial_factory': ['multi_dense', 'multi_conv', 'single_dense', 'single_conv'],  # 4
    'radial_kwargs': [  # 2
        {
            'num_layers': 1,
            'units': 64,
            'activation': 'ssp',
            'kernel_lambda': 0.01,
            'bias_lambda': 0.01
        },
        {
            'num_layers': 2,
            'units': 64,
            'activation': 'ssp',
            'kernel_lambda': 0.01,
            'bias_lambda': 0.01
        }]
}
