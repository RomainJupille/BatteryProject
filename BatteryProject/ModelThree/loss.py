import numpy as np
import tensorflow as tf


# def root_mean_squared_error_tf(y_true, y_pred):
#     y_true = tf.cast(y_true, 'float32')
#     y_pred = tf.cast(y_pred, 'float32')
#     return tf.sqrt(tf.mean(tf.square(y_true - y_pred)))

def root_mean_squared_error(y_true, y_pred):
#     y_true = tf.cast(y_true, 'float32')
#     y_pred = tf.cast(y_pred, 'float32')
#     return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

    y_true = np.array(y_true).reshape(len(y_true), 1)
    y_pred = np.array(y_pred).reshape(len(y_pred), 1)
    return np.sqrt(np.mean(np.square((y_true - y_pred))))
