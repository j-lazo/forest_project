import tensorflow as tf
from .PointNet import tnet, conv_bn, dense_bn

def mixed_input_model(num_outputs, input_shape_iamge=(9,9,60), num_points=1024):

    inputs_points = tf.keras.Input(shape=(num_points, 3))

    x_cloudpoints = tnet(inputs_points, 3)
    x_cloudpoints = conv_bn(x_cloudpoints, 32)
    x_cloudpoints = conv_bn(x_cloudpoints, 64)
    x_cloudpoints = conv_bn(x_cloudpoints, 512)
    x_cloudpoints = tf.keras.layers.GlobalMaxPooling1D()(x_cloudpoints)
    x_cloudpoints = dense_bn(x_cloudpoints, 256)
    x_cloudpoints = tf.keras.layers.Dropout(0.3)(x_cloudpoints)
    x_cloudpoints = dense_bn(x_cloudpoints, 128)
    x_cloudpoints = tf.keras.layers.Dropout(0.3)(x_cloudpoints)
    x_cloudpoints = tf.keras.layers.Flatten()(x_cloudpoints)

    input_raster = tf.keras.Input(shape=input_shape_iamge, name="image_input")
    
    x_raster = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(input_raster)
    x_raster =  tf.keras.layers.ReLU()(x_raster)
    x_raster = tf.keras.layers.BatchNormalization()(x_raster)
    x_raster = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(x_raster)
    x_raster =  tf.keras.layers.ReLU()(x_raster)
    x_raster = tf.keras.layers.BatchNormalization()(x_raster)
    x_raster =  tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x_raster)
    x_raster = tf.keras.layers.Dropout(0.05)(x_raster)
    x_raster = tf.keras.layers.Flatten()(x_raster)


    concat = tf.keras.layers.Concatenate(axis=-1)([x_cloudpoints, x_raster])

    x = tf.keras.layers.Dense(512)(concat)
    x = tf.keras.layers.Dense(256)(concat)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.Dense(32)(x)
    output_layer = tf.keras.layers.Dense(num_outputs)(x)
    
    return tf.keras.Model(inputs=[input_raster, inputs_points], outputs=output_layer, name=f'D_simple_model')