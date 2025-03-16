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


def multi_input_model(num_outputs, input_shape=(9,9,60), center=5):
    input_image = tf.keras.Input(shape=input_shape, name="image_input")
    x = input_image
    radius_1 = 3
    radius_2 = 2
    
    sub_sample_1 =  x[:,center-radius_1:center+radius_1+1, center-radius_1:center+radius_1+1, :]
    sub_sample_2 =  x[:,center-radius_2:center+radius_2+1, center-radius_2:center+radius_2+1, :]

    x = tf.keras.layers.Conv2D(64, (3, 3))(x)
    x =  tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x1 = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(sub_sample_1)
    x1 =  tf.keras.layers.ReLU()(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    c1 = tf.keras.layers.Concatenate(axis=-1)([x, x1])

    x = tf.keras.layers.Conv2D(64, (3, 3))(c1)
    x =  tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x2 = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(sub_sample_2)
    x2 =  tf.keras.layers.ReLU()(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    c2 = tf.keras.layers.Concatenate(axis=-1)([x, x2])

    x = tf.keras.layers.Conv2D(64, (3, 3))(c2)
    x =  tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dropout(0.05)(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64)(x)
    output_layer = tf.keras.layers.Dense(num_outputs)(x)
    return tf.keras.Model(inputs=input_image, outputs=output_layer, name=f'Multi_input_model')
