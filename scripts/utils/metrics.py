import tensorflow as tf

@tf.function
def metric_RootMeanSquaredError(y_true, y_pred):
    y_true = tf.expand_dims(y_true, axis=0)
    y_pred = tf.expand_dims(y_pred, axis=0)
    diff = y_true - y_pred
    size = tf.cast(tf.size(diff), dtype=tf.float32)
    rmse = tf.math.sqrt(tf.math.divide(tf.math.reduce_sum(tf.math.pow(diff, 2)), size))
    return rmse

@tf.function
def metric_MeanSquaredError(y_true, y_pred):
    y_true = tf.expand_dims(y_true, axis=0)
    y_pred = tf.expand_dims(y_pred, axis=0)
    diff = y_true - y_pred
    size = tf.cast(tf.size(diff), dtype=tf.float32)
    mse = tf.math.divide(tf.math.reduce_sum(tf.math.pow(diff, 2)), size)
    return mse

@tf.function
def metric_MeanAbsoluteError(y_true, y_pred):
    y_true = tf.expand_dims(y_true, axis=-1)
    y_pred = tf.expand_dims(y_pred, axis=-1)  
    diff = y_true - y_pred
    size = tf.cast(tf.size(diff), dtype=tf.float32)
    mae = tf.math.divide(tf.math.reduce_sum(tf.math.abs(diff)), size)
    return mae