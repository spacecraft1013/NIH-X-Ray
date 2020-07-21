import keras
def model_generator(hp):
    inputs = keras.layers.Input(shape=(128, 128, 1))
    block1_conv1 = keras.layers.Conv2D(filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=1), kernel_size=hp.Choice('conv_1_kernel', values = [3,5]), activation='relu', padding='same', name='block1_conv1')(inputs)
    block1_conv2 = keras.layers.Conv2D(filters=hp.Int('conv_2_filter', min_value=32, max_value=128, step=1), kernel_size=hp.Choice('conv_2_kernel', values = [3,5]), activation='relu', padding='same', name='block1_conv2')(block1_conv1)
    block1_pooling = keras.layers.MaxPooling2D(hp.Int('pool_1_size', min_value=2, max_value=10, step=1), padding='same', name='block1_pooling')(block1_conv2)

    block2_conv1 = keras.layers.Conv2D(filters=hp.Int('conv_3_filter', min_value=32, max_value=128, step=1), kernel_size=hp.Choice('conv_3_kernel', values = [3,5]), activation='relu', padding='same', name='block2_conv1')(block1_pooling)
    block2_conv2 = keras.layers.Conv2D(filters=hp.Int('conv_4_filter', min_value=32, max_value=128, step=1), kernel_size=hp.Choice('conv_4_kernel', values = [3,5]), activation='relu', padding='same', name='block2_conv2')(block2_conv1)
    block2_pooling = keras.layers.MaxPooling2D(hp.Int('pool_2_size', min_value=2, max_value=10, step=1), padding='same', name='block2_pooling')(block2_conv2)

    block3_conv1 = keras.layers.Conv2D(filters=hp.Int('conv_5_filter', min_value=32, max_value=128, step=1), kernel_size=hp.Choice('conv_5_kernel', values = [3,5]), activation='relu', padding='same', name='block3_conv1')(block2_pooling)
    block3_conv2 = keras.layers.Conv2D(filters=hp.Int('conv_6_filter', min_value=32, max_value=128, step=1), kernel_size=hp.Choice('conv_6_kernel', values = [3,5]), activation='relu', padding='same', name='block3_conv2')(block3_conv1)
    block3_conv3 = keras.layers.Conv2D(filters=hp.Int('conv_7_filter', min_value=32, max_value=128, step=1), kernel_size=hp.Choice('conv_7_kernel', values = [3,5]), activation='relu', padding='same', name='block3_conv3')(block3_conv2)
    block3_pooling = keras.layers.MaxPooling2D(hp.Int('pool_3_size', min_value=2, max_value=10, step=1), padding='same', name='block3_pooling')(block3_conv3)

    block4_conv1 = keras.layers.Conv2D(filters=hp.Int('conv_8_filter', min_value=32, max_value=128, step=1), kernel_size=hp.Choice('conv_8_kernel', values = [3,5]), activation='relu', padding='same', name='block4_conv1')(block3_pooling)
    block4_conv2 = keras.layers.Conv2D(filters=hp.Int('conv_9_filter', min_value=32, max_value=128, step=1), kernel_size=hp.Choice('conv_9_kernel', values = [3,5]), activation='relu', padding='same', name='block4_conv2')(block4_conv1)
    block4_conv3 = keras.layers.Conv2D(filters=hp.Int('conv_10_filter', min_value=32, max_value=128, step=1), kernel_size=hp.Choice('conv_10_kernel', values = [3,5]), activation='relu', padding='same', name='block4_conv3')(block4_conv2)
    block4_conv4 = keras.layers.Conv2D(filters=hp.Int('conv_11_filter', min_value=32, max_value=128, step=1), kernel_size=hp.Choice('conv_11_kernel', values = [3,5]), activation='relu', padding='same', name='block4_conv4')(block4_conv3)

    flatten = keras.layers.Flatten(name='flatten')(block4_conv4)
    
    dense1 = keras.layers.Dense(hp.Int('dense_1_units', min_value=1, max_value=256, step=1), activation='relu', name='fc1')(flatten)
    dense2 = keras.layers.Dense(hp.Int('dense_2_units', min_value=1, max_value=256, step=1), activation='relu', name='fc2')(dense1)
    output = keras.layers.Dense(15, activation='softmax', name='output')(dense2)

    model = keras.models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=hp.Choice('optimizer', values=['adam', 'adadelta', 'sgd']), loss='binary_crossentropy', metrics=['accuracy'])

    return model