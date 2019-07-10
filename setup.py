from setuptools import setup, find_packages

setup(
    name='TFN-Layers',
    author='Riley Jackson',
    author_email='rjjackson@upei.ca',
    version='0.0.0',
    description='Keras layers for rotationally equivariant Convolutional Tensor Field Networks',
    packages=find_packages(),
    install_requires=['tensorflow', 'numpy', 'pytest']
)
