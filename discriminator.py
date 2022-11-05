import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            # Here in_channels is multiplied by two because we're going to send in both images concatenated in the channel dimension.
            # aka we'll be having 6 channels. (3 for each image)
            nn.Conv2d(in_channels*2, features[0], kernel_size=4,
                      stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = features[0]

        for feature in features[1:]:
            layers.append(
                downsampling_conv(in_channels, feature,
                                  stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            )
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)


# Convolution segment for the discriminator(downsampling)
# Convolution -> BatchNorm -> leakyRelu
class downsampling_conv(nn.Module):
    def __init__(self, in_planes, out_planes, ks=4, stride=2, pad=1, bn=True):
        super().__init__()
        # stores if the batch norm should be applied
        self.bn = bn

        self.conv_downsample = nn.Conv2d(
            in_planes, out_planes, ks, stride, pad, padding_mode="reflect")
        self.batch_norm = nn.BatchNorm2d(out_planes)

        LEAKY_RELU_SLOPE = 0.2
        self.leaky_relu = nn.LeakyReLU(LEAKY_RELU_SLOPE)

    def forward(self, x):
        # Run through convolution then BatchNorm then leaky ReLU

        x = self.conv_downsample(x)

        if self.bn:
            x = self.batch_norm(x)

        x = self.leaky_relu(x)

        return x
