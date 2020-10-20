"""
Package containing Tensorfield Network tf.keras layers built using TF 2.0
"""
import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
logging.getLogger("tensorflow").setLevel(logging.FATAL)

__version__ = "2.5.0"
__author__ = "Riley Jackson"
__email__ = "rjjackson@upei.ca"
__description__ = (
    "Package containing Tensor Field Network tf.keras layers built using Tensorflow 2"
)
