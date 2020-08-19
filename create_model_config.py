import keras

densenet = keras.applications.DenseNet201(input_shape=(256, 256, 1), weights=None, classes=15)

regularizer = keras.regularizers.l1_l2(l1=0.01, l2=0.01)

for layer in densenet.layers:
    for attr in ['kernel_regularizer']:
        if hasattr(layer, attr):
            setattr(layer, attr, regularizer)

model_config = densenet.to_json()

with open("data/models/model_config.json", "a") as config_file:
    config_file.write(model_config)