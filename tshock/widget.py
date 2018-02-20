"""    
tshock:widget
author:@elvinlife
time:since 2018/2/7
"""
import tensorflow as tf
from .initializer import *

class Widget(object):
    def __init__(self, name=None):
        """
        Build the widget.
        :param name:
        """
        self._name = name
        self._scope = tf.get_variable_scope().name
        self._prefix = self._scope + '/'
        if name == None:
            self._build()
        else:
            self._prefix = self._prefix + self._name + '/'
            with tf.variable_scope(self._name):
                self._build()

    def _build(self):
        """
        All inherited class must override this method to build itself.
        Can only be called once.
        :return:None
        """
        raise NotImplementedError("{} is an abstract widget".format(self._name))

    def setup(self, *args, **kwargs):
        """
        Add this widget into your origin TF graph
        :return:None
        """
        raise NotImplementedError("{} is an abstract widget".format(self._name))

    @property
    def name(self):
        return self._name

    @property
    def scope(self):
        return self._scope

    @property
    def global_variable(self):
        return

class Linear(Widget):
    def __init__(self,
                 name = None,
                 input_size = None,
                 output_size = None,
                 with_bias=True,
                 trainable = True,
                 weight_initializer=RandomNormal(),
                 bias_initializer=ZerosInit(),
                 ):
        """
        :param input_size:
        :param output_size:
        :param with_bias:
        :param weight_initializer:
        :param bias_initializer:
        """
        self._input_size = input_size
        self._output_size = output_size
        self._with_bias = with_bias
        self._weight_initializer = weight_initializer
        self._bias_initializer = bias_initializer
        self._trainable = trainable
        print(weight_initializer)
        super(Linear, self).__init__(name)

    def _build(self):
        self._w = tf.Variable(
            initial_value=self._weight_initializer.build(
                (self._input_size, self._output_size),
                tf.float32
            ),
            trainable=self._trainable,
            name='w'
        )
        self._b = tf.Variable(
            initial_value=self._bias_initializer.build(
                (1, self._output_size),
                tf.float32
            ),
            trainable=self._trainable,
            name='b'
        ) if self._with_bias else None

    def setup(self, x):
        y = tf.matmul(x, self._w)
        y = y + self._b if self._with_bias else y
        return y