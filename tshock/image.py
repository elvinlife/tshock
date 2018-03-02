"""    
tshock:image
author:@elvinlife
time:since 2018/3/1
"""
import tensorflow as tf
from .type import float_default

def to_dense(images, mean=127.5, std=128):
    return (tf.cast(images, float_default)-mean)/std

def sub_sample(
        images,
        patch_height,
        patch_width,
        stride_height = None,
        stride_width = None,
        with_padding = True,
        fuct=tf.reduce_mean
):
    stride_height = patch_height if stride_height is None else stride_height
    stride_width = patch_width if stride_width is None else stride_width
    patches = tf.extract_image_patches(
        images=images,
        ksizes=(1, patch_height, patch_width, 1),
        strides=(1, stride_height, stride_width, 1),
        rates=(1,1,1,1),
        padding='SAME' if with_padding else 'VALID'
    )
    channel = images.shape[-1]
    new_shape = [*patches.shape[:-1], patches.shape[-1]//channel, channel]
    new_shape[0] = -1
    return fuct(
        tf.reshape(patches, shape=new_shape),
        axis=-2
    )