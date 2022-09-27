import keras
import torch.nn


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

class NIHXrayModel(nn.Module):
    def __init__(self, model_config: dict, input_size: tuple[int, int, int, int], num_classes: int = 14):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes

    def forward(self, x):
        pass

    def build_model(self):
        layers = []
        for i in range(self.num_blocks):
            layers.append(self.conv_block(self.num_filters, self.kernel_size, self.padding, self.stride))

    def conv_block(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
