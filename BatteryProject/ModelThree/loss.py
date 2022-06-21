import numpy as np
import tensorflow as tf


def root_mean_squared_error(y_true, y_pred):
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(y_pred, 'float32')
    # y_pred = y_pred.numpy().astype('float32')
    return tf.sqrt(tf.math.reduce_mean(tf.square(y_pred - y_true)))
