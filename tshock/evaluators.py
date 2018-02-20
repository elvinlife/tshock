"""    
tshock:evaluators
author:@elvinlife
time:since 2018/2/18
"""
import tensorflow as tf

def safe_log(t):
    return tf.log(t+1e-8)

def categorical_cross_entropy_loss(y_true, y_pred, with_false=True, f_reduce = tf.reduce_mean):
    loss = - tf.reduce_sum((y_true * safe_log(y_pred)), axis=1)
    if with_false:
        loss_false = - tf.reduce_sum((1-y_true) * safe_log(1-y_pred), axis=1)
        loss += loss_false
    return f_reduce(loss)