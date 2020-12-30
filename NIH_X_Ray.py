import datetime
import os

import numpy as np
import keras
from keras.callbacks import ModelCheckpoint, TensorBoard

from multithreaded_preprocessing import PreprocessImages

model_save_name = "densenet201_kld"
epochs = 250
image_size = 256
batch_size = 32

print("Importing Arrays")
if not os.path.exists(f"data/arrays/X_train_{image_size}.npy"):
    print("Arrays not found, generating...")
    preprocessor = PreprocessImages("F:/Datasets/NIH X-Rays/data", image_size)
    (X_train, y_train), (X_test, y_test) = preprocessor()

else:
    X_train = np.load(open(f"data/arrays/X_train_{image_size}.npy", "rb"))
    y_train = np.load(open(f"data/arrays/y_train_{image_size}.npy", "rb"))
    X_test = np.load(open(f"data/arrays/X_test_{image_size}.npy", "rb"))
    y_test = np.load(open(f"data/arrays/y_test_{image_size}.npy", "rb"))

# tuner = kt.BayesianOptimization(model_generator,
#                                 objective='val_accuracy', max_trials=500,
#                                 project_name="NIH X-Ray Model")

# tuner.search(X_train, y_train, epochs=5, validation_split=0.1)

# best_model = tuner.get_best_models(1)[0]
# print(best_model.summary())

with open("data/models/model_config.json", "r") as model_config:
    densenet = keras.models.model_from_json(model_config.read())

densenet.compile(optimizer=keras.optimizers.Adam(lr=1e-6),
                 loss='kullback_leibler_divergence', metrics=['accuracy'])

log_dir = f'data/logs/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}\
            -{model_save_name}'
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint_path = "data/checkpoints/" + model_save_name + "-{epoch:03d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
densenet.save_weights(checkpoint_path.format(epoch=0))
cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                              save_weights_only=True, verbose=1)

callbacks = [tensorboard_callback, cp_callback]
densenet.fit(X_train, y_train, batch_size=32, epochs=epochs,
             validation_data=(X_test, y_test), callbacks=callbacks)

test_loss, test_acc = densenet.evaluate(X_test, y_test)
print(f'\nTest Accuracy: {test_acc}\nTest Loss: {test_loss}')

densenet.save(f'data/models/{model_save_name}.h5')
