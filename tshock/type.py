"""    
tshock:private
author:@elvinlife
time:since 2018/2/18
"""
import collections
import tensorflow as tf

def _make_tuple(obj):
    if isinstance(obj, collections.Iterable):
        return tuple(obj)
    elif obj == None:
        return ()
    else:
        return obj,

float_default = tf.float32
int_default = tf.int32
