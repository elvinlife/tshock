"""
tshock:initializer
author:@elvinlife
time:since 2018/2/7
"""
import tensorflow as tf

class Initializer(object):
    """
    Initializer base class: all initializers inherit from this class.
    """

    def build(self, shape, dtype, name=None, seed=None):
        raise NotImplementedError()

class ZerosInit(Initializer):
    def build(self, shape, dtype, name=None, seed=None):
        return tf.zeros(shape, dtype, name)

class RandomNormal(Initializer):
    def __init__(self, mean=0., stddev=0.05):
        self._mean = mean
        self._stddev = stddev

    @property
    def mean(self):
        return self._mean

    @property
    def stddev(self):
        return self._stddev

    def build(self, shape, dtype, name=None, seed=None):
        return tf.random_normal(
            shape = shape,
            mean = self._mean,
            stddev = self._stddev,
            dtype = dtype,
            seed = seed,
            name = name
        )