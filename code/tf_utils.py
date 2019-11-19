import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope, arg_scope
from tensorflow.contrib.layers import variance_scaling_initializer

'''
actnorm function: the implementation taken from https://github.com/openai/glow
'''


def allreduce_sum(x):
    if hvd.size() == 1:
        return x
    return hvd.mpi_ops._allreduce(x)


def allreduce_mean(x):
    x = allreduce_sum(x) / hvd.size()
    return x


def default_initial_value(shape, std=0.05):
    return tf.random_normal(shape, 0., std)


def default_initializer(std=0.05):
    return tf.random_normal_initializer(0., std)


def int_shape(x):
    if str(x.get_shape()[0]) != '?':
        return list(map(int, x.get_shape()))
    return [-1]+list(map(int, x.get_shape()[1:]))


@add_arg_scope
def get_variable_ddi(name, shape, initial_value, dtype=tf.float32, init=False, trainable=True):
    w = tf.get_variable(name, shape, dtype, None, trainable=trainable)
    if init:
        w = w.assign(initial_value)
        with tf.control_dependencies([w]):
            return w
    return w

@add_arg_scope
def actnorm(name, x, scale=1., logdet=None, logscale_factor=3., batch_variance=False, reverse=False, init=False, trainable=True):
    if arg_scope([get_variable_ddi], trainable=trainable):
        if not reverse:
            x = actnorm_center(name+"_center", x, reverse)
            x = actnorm_scale(name+"_scale", x, scale, logdet,
                              logscale_factor, batch_variance, reverse, init)
            if logdet != None:
                x, logdet = x
        else:
            x = actnorm_scale(name + "_scale", x, scale, logdet,
                              logscale_factor, batch_variance, reverse, init)
            if logdet != None:
                x, logdet = x
            x = actnorm_center(name+"_center", x, reverse)
        if logdet != None:
            return x, logdet
        return x

@add_arg_scope
def actnorm_center(name, x, reverse=False):
    shape = x.get_shape()
    with tf.variable_scope(name):
        assert len(shape) == 2 or len(shape) == 4
        if len(shape) == 2:
            x_mean = tf.reduce_mean(x, [0], keepdims=True)
            b = get_variable_ddi(
                "b", (1, int_shape(x)[1]), initial_value=-x_mean)
        elif len(shape) == 4:
            x_mean = tf.reduce_mean(x, [0, 1, 2], keepdims=True)
            b = get_variable_ddi(
                "b", (1, 1, 1, int_shape(x)[3]), initial_value=-x_mean)

        if not reverse:
            x += b
        else:
            x -= b

        return x

@add_arg_scope
def actnorm_scale(name, x, scale=1., logdet=None, logscale_factor=3., batch_variance=False, reverse=False, init=False, trainable=True):
    shape = x.get_shape()
    with tf.variable_scope(name), arg_scope([get_variable_ddi], trainable=trainable):
        assert len(shape) == 2 or len(shape) == 4
        if len(shape) == 2:
            x_var = tf.reduce_mean(x**2, [0], keepdims=True)
            logdet_factor = 1
            _shape = (1, int_shape(x)[1])

        elif len(shape) == 4:
            x_var = tf.reduce_mean(x**2, [0, 1, 2], keepdims=True)
            logdet_factor = int(shape[1])*int(shape[2])
            _shape = (1, 1, 1, int_shape(x)[3])

        if batch_variance:
            x_var = tf.reduce_mean(x**2, keepdims=True)

        if init and False:
            # MPI all-reduce
            x_var = allreduce_mean(x_var)
            # Somehow this also slows down graph when not initializing
            # (it's not optimized away?)

        if True:
            logs = get_variable_ddi("logs", _shape, initial_value=tf.log(
                scale/(tf.sqrt(x_var)+1e-6))/logscale_factor)*logscale_factor
            if not reverse:
                x = x * tf.exp(logs)
            else:
                x = x * tf.exp(-logs)
        else:
            # Alternative, doesn't seem to do significantly worse or better than the logarithmic version above
            s = get_variable_ddi("s", _shape, initial_value=scale /
                                 (tf.sqrt(x_var) + 1e-6) / logscale_factor)*logscale_factor
            logs = tf.log(tf.abs(s))
            if not reverse:
                x *= s
            else:
                x /= s

        if logdet != None:
            dlogdet = tf.reduce_sum(logs) * logdet_factor
            if reverse:
                dlogdet *= -1
            return x, logdet + dlogdet

        return x
