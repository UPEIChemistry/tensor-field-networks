from setuptools import setup, find_packages

setup(
    name='tfn-Layers',
    author='Riley Jackson',
    author_email='rjjackson@upei.ca',
    version='0.0.0',
    description='Keras layers for rotationally equivariant Convolutional Tensor Field Networks',
    packages=find_packages(),
    install_requires=['tensorflow==2.0.0-beta1', 'numpy']
)
