import os
import re
import socket
from pathlib import Path
from shutil import rmtree

import pytest
from pytest import fixture


test_dir = Path(os.path.abspath(os.path.dirname(__file__))) / 'test_jobs'
test_model_path = test_dir / 'test_model.h5'

if socket.gethostname() == 'tater':
    os.environ['DATADIR'] = '/mnt/fast-data/riley/data'
else:
    os.environ['DATADIR'] = '/home/riley/dev/python/data'


def clear_directories():
    rmtree(test_dir / 'logs', ignore_errors=True)
    rmtree(test_dir / 'sacred_storage', ignore_errors=True)
    rmtree(test_dir / 'tuner_storage', ignore_errors=True)
    try:
        for f in os.listdir(test_dir):
            if re.search(r'.*\.h5', f):
                os.remove(test_dir / f)
        os.remove(test_model_path)
    except IsADirectoryError:
        rmtree(test_model_path, ignore_errors=True)
    except FileNotFoundError:
        pass


@fixture(scope='session', autouse=True)
def clear_logdirs(request):
    request.addfinalizer(clear_directories)


@pytest.fixture(scope='session')
def model():
    return str(test_model_path)


@pytest.fixture
def run_config():
    return {
        'epochs': 2,
        'test': False,
        'save_model': True,
        'use_strategy': False,
        'select_few': 50,
        'run_eagerly': True,
        'model_path': str(test_model_path)
    }


@pytest.fixture
def builder_config():
    return {
        'dynamic': True
    }


@pytest.fixture
def factory_config():
    return {
        'run_eagerly': False,
        'dynamic': False
    }


@pytest.fixture
def tuner_config():
    return {
        'objective': 'loss'
    }


@pytest.fixture
def architecture_search():
    return {
        'si_units': {
            'type': 'choice', 'kwargs': {
                'values': [8, 16]
            }
        }
    }
