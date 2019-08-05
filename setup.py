from setuptools import setup, find_packages

setup(
    name='tfn-layers',
    author='Riley Jackson',
    author_email='rjjackson@upei.ca',
    version='1.0.0',
    description='Keras layers for rotationally equivariant Convolutional Tensor Field Networks',
    packages=find_packages(),
    install_requires=[
        'tensorflow-gpu==2.0.0b1',
        'numpy'
    ],
    extras_require={
        'tests': ['pytest'],
        'tensorflow': ['tensorflow==2.0.0b1'],
        'h5py': ['h5py']
    }
)
