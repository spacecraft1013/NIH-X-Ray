import keras
def model(shape_value):
    inputs = keras.layers.Input(shape=(shape_value, shape_value, 1))
    block1_conv1 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    block1_conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(block1_conv1)
    block1_pooling = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pooling')(block1_conv2)

    block2_conv1 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(block1_pooling)
    block2_conv2 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(block2_conv1)
    block2_pooling = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pooling')(block2_conv2)

    block3_conv1 = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv1')(block2_pooling)
    block3_conv2 = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv2')(block3_conv1)
    block3_conv3 = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv3')(block3_conv2)
    block3_pooling = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pooling')(block3_conv3)

    block4_conv1 = keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv1')(block3_pooling)
    block4_conv2 = keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv2')(block4_conv1)
    block4_conv3 = keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv3')(block4_conv2)
    block4_conv4 = keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv4')(block4_conv3)

    flatten = keras.layers.Flatten(name='flatten')(block4_conv4)
    
    dense1 = keras.layers.Dense(256, activation='relu', name='fc1')(flatten)
    dense2 = keras.layers.Dense(128, activation='relu', name='fc2')(dense1)
    output = keras.layers.Dense(15, activation='softmax', name='output')(dense2)

    model = keras.models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mae'])

    return model