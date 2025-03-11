import tensorflow as tf

def simple_regressor(num_outputs, input_shape=(9,9,60)):
    input_image = tf.keras.Input(shape=input_shape, name="image_input")

    x = input_image
    x = tf.keras.layers.Conv2D(60, (3, 3), padding="same")(x)
    x =  tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024*2, activation='relu')(x)

    #x = tf.keras.layers.Dense(1024)(x)
    #x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    #x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    #x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    output_layer = tf.keras.layers.Dense(num_outputs)(x)
    return tf.keras.Model(inputs=input_image, outputs=output_layer, name=f'simple_model')


def D_model(num_outputs, input_shape=(9,9,60)):
    input_image = tf.keras.Input(shape=input_shape, name="image_input")
    x = input_image
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(x)
    x =  tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(x)
    x =  tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x =  tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
    x = tf.keras.layers.Dropout(0.05)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(254)(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.Dense(32)(x)
    output_layer = tf.keras.layers.Dense(num_outputs)(x)
    return tf.keras.Model(inputs=input_image, outputs=output_layer, name=f'D_simple_model')