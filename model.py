import keras
def model(version, shape_value):
    if version == "v1":
        inputs = keras.layers.Input(shape=(shape_value, shape_value, 1))
        conv_1 = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        conv_2 = keras.layers.Conv2D(64, (3, 3), activation='relu')(conv_1)
        conv_3 = keras.layers.Conv2D(128, (3, 3), activation='relu')(conv_2)
        pooling_1 = keras.layers.MaxPooling2D((2, 2))(conv_3)
        conv_4 = keras.layers.Conv2D(64, (3, 3), activation='relu')(pooling_1)
        conv_5 = keras.layers.Conv2D(64, (3, 3), activation='relu')(conv_4)
        conv_6 = keras.layers.Conv2D(32, (3, 3), activation='relu')(conv_5)
        pooling_2 = keras.layers.GlobalAveragePooling2D()(conv_6)
        dense_1 = keras.layers.Dense(2048, activation='relu')(pooling_2)
        dense_2 = keras.layers.Dense(1024, activation='relu')(dense_1)
        dropout_1 = keras.layers.Dropout(0.2)(dense_2)
        dense_3 = keras.layers.Dense(512, activation='relu')(dropout_1)
        dropout_2 = keras.layers.Dropout(0.2)(dense_3)
        dense_4 = keras.layers.Dense(128, activation='relu')(dropout_2)
        dropout_3 = keras.layers.Dropout(0.2)(dense_4)
        dense_5 = keras.layers.Dense(64, activation='relu')(dropout_3)
        dropout_4 = keras.layers.Dropout(0.2)(dense_5)
        dense_6 = keras.layers.Dense(32, activation='relu')(dropout_4)
        output = keras.layers.Dense(15, activation='sigmoid')(dense_6)

        model = keras.models.Model(inputs=inputs, outputs=output)

    if version == "v2":
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
        block4_pooling = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pooling')(block4_conv4)

        block5_conv1 = keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv1')(block4_pooling)
        block5_conv2 = keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv2')(block5_conv1)
        block5_conv3 = keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv3')(block5_conv2)
        block5_conv4 = keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv4')(block5_conv3)
        block5_conv5 = keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv5')(block5_conv4)
        
        flatten = keras.layers.Flatten(name='flatten')(block5_conv5)
        
        dense1 = keras.layers.Dense(256, activation='relu', name='fc1')(flatten)
        dense2 = keras.layers.Dense(128, activation='relu', name='fc2')(dense1)
        output = keras.layers.Dense(15, activation='softmax', name='output')(dense2)

        model = keras.models.Model(inputs=inputs, outputs=output)

    if version == "v3":
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

    return model