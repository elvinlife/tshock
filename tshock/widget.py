"""    
tshock:widget
author:@elvinlife
time:since 2018/2/7
"""
from .initializer import *
from .type import *
from .type import _make_tuple

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
                 input_size,
                 output_size,
                 with_bias=True,
                 trainable = True,
                 weight_initializer=RandomNormal(),
                 bias_initializer=ZerosInit(),
                 name=None,
                 ):
        """
        :param input_size: int
        :param output_size: int
        :param with_bias: boolean
        :param weight_initializer: ts.Initializer
        :param bias_initializer: ts.Initializer
        """
        self._input_size = input_size
        self._output_size = output_size
        self._with_bias = with_bias
        self._weight_initializer = weight_initializer
        self._bias_initializer = bias_initializer
        self._trainable = trainable
        super(Linear, self).__init__(name)

    def _build(self):
        self._w = tf.Variable(
            initial_value=self._weight_initializer.build(
                (self._input_size, self._output_size),
                float_default
            ),
            trainable=self._trainable,
            name='w'
        )
        self._b = tf.Variable(
            initial_value=self._bias_initializer.build(
                (1, self._output_size),
                float_default
            ),
            trainable=self._trainable,
            name='b'
        ) if self._with_bias else None

    def setup(self, x):
        y = tf.matmul(x, self._w)
        y = y + self._b if self._with_bias else y
        return y

class Conv2d(Widget):
    def __init__(self,
                 input_shape,
                 out_channels,
                 filter_height=3,
                 filter_width=3,
                 stride_height=1,
                 stride_width=1,
                 with_padding=False,
                 weight_initializer=RandomNormal(),
                 bias_initializer=ZerosInit(),
                 dtype=float_default,
                 name=None,
                 ):
        self._input_shape = _make_tuple(input_shape)
        assert len(self._input_shape) == 3,'2D Convoluted Widget input_size need 3 ' \
                                           'elements(image_height, image_width, channels)'
        self._in_height = self._input_shape[0]
        self._in_width = self._input_shape[1]
        self._in_channels = self._input_shape[2]
        self._out_channels = out_channels
        self._filter_height = filter_height
        self._filter_width = filter_width
        self._stride_height = stride_height
        self._stride_width = stride_width
        self._with_padding = with_padding
        self._weight_initializer=weight_initializer
        self._bias_initializer=bias_initializer
        self._dtype=dtype
        self._name=name
        super(Conv2d, self).__init__(name)

    def _build(self):
        self._w = tf.Variable(
            initial_value=self._weight_initializer.build(
                shape=(self._filter_height, self._filter_width, self._in_channels, self._out_channels),
                dtype=self._dtype,
                name='w'
            )
        )
        self._b = tf.Variable(
            initial_value=self._bias_initializer.build(
                shape=(self._out_channels,),
                dtype=self._dtype,
                name='b'
            )
        )

    def setup(self, batch):
        if len(batch.shape) == 3:
            batch = tf.expand_dims(batch, 3)
        return tf.nn.conv2d(
            input=batch,
            filter=self._w,
            strides=[1, self._stride_height, self._stride_width, 1],
            padding='SAME' if self._with_padding else 'VALID'
        )+self._b