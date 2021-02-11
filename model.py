import keras


def model_generator_keras(hp):

    model = keras.models.Sequential()

    model.add(keras.layers.Input(shape=(128, 128, 1)))
    
    for i in range(hp.Int('3blocks', min_value=0, max_value=150, step=1)):
        model.add(keras.layers.Conv2D(filters=hp.Int('3block_conv_filter', min_value=32, max_value=128, step=1), kernel_size=hp.Choice('3block_conv_kernel', values = [3,5]), activation='relu', padding='same'))
        model.add(keras.layers.Conv2D(filters=hp.Int('3block_conv_filter', min_value=32, max_value=128, step=1), kernel_size=hp.Choice('3block_conv_kernel', values = [3,5]), activation='relu', padding='same'))
        model.add(keras.layers.Conv2D(filters=hp.Int('3block_conv_filter', min_value=32, max_value=128, step=1), kernel_size=hp.Choice('3block_conv_kernel', values = [3,5]), activation='relu', padding='same'))
        model.add(keras.layers.MaxPooling2D(hp.Int('3block_pool_size', min_value=2, max_value=10, step=1), padding='same'))

    for i in range(hp.Int('4blocks', min_value=0, max_value=150, step=1)):
        model.add(keras.layers.Conv2D(filters=hp.Int('4block_conv_filter', min_value=32, max_value=128, step=1), kernel_size=hp.Choice('4block_conv_kernel', values = [3,5]), activation='relu', padding='same'))
        model.add(keras.layers.Conv2D(filters=hp.Int('4block_conv_filter', min_value=32, max_value=128, step=1), kernel_size=hp.Choice('4block_conv_kernel', values = [3,5]), activation='relu', padding='same'))
        model.add(keras.layers.Conv2D(filters=hp.Int('4block_conv_filter', min_value=32, max_value=128, step=1), kernel_size=hp.Choice('4block_conv_kernel', values = [3,5]), activation='relu', padding='same'))
        model.add(keras.layers.Conv2D(filters=hp.Int('4block_conv_filter', min_value=32, max_value=128, step=1), kernel_size=hp.Choice('4block_conv_kernel', values = [3,5]), activation='relu', padding='same'))
        model.add(keras.layers.MaxPooling2D(hp.Int('4block_pool_size', min_value=2, max_value=10, step=1), padding='same'))

    for i in range(hp.Int('5blocks', min_value=0, max_value=150, step=1)):
        model.add(keras.layers.Conv2D(filters=hp.Int('5block_conv_filter', min_value=32, max_value=128, step=1), kernel_size=hp.Choice('5block_conv_kernel', values = [3,5]), activation='relu', padding='same'))
        model.add(keras.layers.Conv2D(filters=hp.Int('5block_conv_filter', min_value=32, max_value=128, step=1), kernel_size=hp.Choice('5block_conv_kernel', values = [3,5]), activation='relu', padding='same'))
        model.add(keras.layers.Conv2D(filters=hp.Int('5block_conv_filter', min_value=32, max_value=128, step=1), kernel_size=hp.Choice('5block_conv_kernel', values = [3,5]), activation='relu', padding='same'))
        model.add(keras.layers.Conv2D(filters=hp.Int('5block_conv_filter', min_value=32, max_value=128, step=1), kernel_size=hp.Choice('5block_conv_kernel', values = [3,5]), activation='relu', padding='same'))
        model.add(keras.layers.Conv2D(filters=hp.Int('5block_conv_filter', min_value=32, max_value=128, step=1), kernel_size=hp.Choice('5block_conv_kernel', values = [3,5]), activation='relu', padding='same'))
        model.add(keras.layers.MaxPooling2D(hp.Int('5block_pool_size', min_value=2, max_value=10, step=1), padding='same'))

    model.add(keras.layers.Flatten(name='flatten'))
    
    for i in range(hp.Int('denselayers', min_value=1, max_value=100, step=1)):
        model.add(keras.layers.Dense(hp.Int('dense_units', min_value=1, max_value=256, step=1), activation='relu'))
    
    model.add(keras.layers.Dense(14, activation=hp.Choice('activation_function', values=['softmax', 'sigmoid']), name='output'))

    model.compile(optimizer=hp.Choice('optimizer', values=['adam', 'adadelta', 'sgd']), loss='categorical_crossentropy', metrics=['accuracy'])

    return model
