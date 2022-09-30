import torch
import torch.nn as nn

from utils import Config


class NIHXRayModel(nn.Module):
    def __init__(self, model_config: Config, input_size: tuple[int, int, int, int], num_classes: int = 14):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.config = model_config

        self.layers = None

        self.build_model()

    def forward(self, x) -> torch.Tensor:
        assert self.layers is not None, "Model must be built before utilization"
        input_tensor = x
        if self.config.structure == 'dense':
            conns = [x]
        for layer in self.layers:
            x = layer(x)
            if self.config.structure == 'dense':
                conns.append(x)
                x = torch.sum(conns)
        if self.config.structure == 'res':
            x += input_tensor
        x = self.output_block(x)
        return x

    def build_model(self) -> None:
        layers = []
        for i in range(self.num_blocks):
            layers.append(self.conv_block(self.config.num_filters,
                          self.config.num_filters, self.config.kernel_size, self.config.stride))

        self.layers = layers

    def input_block(self, out_channels: int) -> nn.Module:
        return nn.Conv2d(self.input_size[1], out_channels, self.config.input_filter_size)

    def conv_block(self, in_channels, out_channels, kernel_size, stride) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size, stride, padding='same'),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size, stride, padding='same'),
            nn.ReLU()
        )

    def output_block(self, in_channels) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_channels*self.input_size[1]*self.input_size[2], self.config.classification_hidden_features),
            nn.Linear(self.config.classification_hidden_features, self.num_classes)
        )
