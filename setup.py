from setuptools import setup, find_packages

setup(
    name='tfn-layers',
    author='Riley Jackson',
    author_email='rjjackson@upei.ca',
    version='1.0.0',
    description='Keras layers for rotationally equivariant Convolutional Tensor Field Networks',
    packages=find_packages(),
    extras_require={
        'tests': ['pytest'],
        'tf': ['tensorflow-gpu==2.0.0b1']
    }
)
