import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # input:
        self.input_conv = nn.Conv2d(
            in_channels, 64, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        # encoder:
        # C64-C128-C256-C512-C512-C512-C512-C512
        self.enc_c64_128 = enc_downsampling_conv(64, 128, bn=False)
        self.enc_c128_256 = enc_downsampling_conv(128, 256)
        self.enc_c256_512 = enc_downsampling_conv(256, 512)
        self.enc_c512_512 = enc_downsampling_conv(512, 512)

        # decoder
        # CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
        # Note this is the U-Net version
        self.dec_cd512_512 = dec_dropout_upsampling_conv(512, 512)
        self.dec_cd1024_512 = dec_dropout_upsampling_conv(1024, 512)

        self.dec_c1024_512 = dec_upsampling_conv(1024, 512)
        self.dec_c1024_256 = dec_dropout_upsampling_conv(1024, 256)
        self.dec_c512_128 = dec_upsampling_conv(512, 128)
        self.dec_c256_64 = dec_upsampling_conv(256, 64)

        # output
        self.output_conv = nn.ConvTranspose2d(
            128, out_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, out):

        # INPUT
        out = self.input_conv(out)
        out = self.leaky_relu(out)
        skip_l1 = out

        # ENCODE
        # down 1:
        out = self.enc_c64_128(out)
        skip_l2 = out
        # down 2:
        out = self.enc_c128_256(out)
        skip_l3 = out
        # down 3:
        out = self.enc_c256_512(out)
        skip_l4 = out
        # down 4:
        out = self.enc_c512_512(out)
        skip_l5 = out
        # down 5:
        out = self.enc_c512_512(out)
        skip_l6 = out
        # down 6:
        out = self.enc_c512_512(out)
        skip_l7 = out

        # bottleneck:
        out = self.enc_c512_512(out)

        # DECODE
        # up 1:
        out = self.dec_cd512_512(out)
        # up 2:
        out = torch.cat((out, skip_l7), 1)
        out = self.dec_cd1024_512(out)
        # up 3:
        out = torch.cat((out, skip_l6), 1)
        out = self.dec_cd1024_512(out)
        # up 4:
        out = torch.cat((out, skip_l5), 1)
        out = self.dec_c1024_512(out)
        # up 5:
        out = torch.cat((out, skip_l4), 1)
        out = self.dec_c1024_256(out)
        # up 6:
        out = torch.cat((out, skip_l3), 1)
        out = self.dec_c512_128(out)
        # up 7:
        out = torch.cat((out, skip_l2), 1)
        out = self.dec_c256_64(out)

        # OUTPUT
        out = torch.cat((out, skip_l1), 1)
        out = self.output_conv(out)
        out = self.tanh(out)

        return out


# Convolution segment for encoding (downsampling)
# Convolution -> BatchNorm -> leakyRelu
class enc_downsampling_conv(nn.Module):
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

# Convolution segment for decoding (upsampling) with dropout
# Convolution -> BatchNorm -> Dropout -> Relu


class dec_dropout_upsampling_conv(nn.Module):
    def __init__(self, in_planes, out_planes, ks=4, stride=2, pad=1):
        super().__init__()

        self.conv_upsample = nn.ConvTranspose2d(
            in_planes, out_planes, ks, stride, pad)

        self.batch_norm = nn.BatchNorm2d(out_planes)

        DROPOUT_RATE = 0.5
        self.dropout = nn.Dropout2d(DROPOUT_RATE)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Run through convolution then dropout then ReLU
        x = self.conv_upsample(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.relu(x)

        return x

# Convolution segment for decoding (upsampling) without dropout
# Convolution -> BatchNorm -> Relu


class dec_upsampling_conv(nn.Module):
    def __init__(self, in_planes, out_planes, ks=4, stride=2, pad=1):
        super().__init__()

        self.conv_upsample = nn.ConvTranspose2d(
            in_planes, out_planes, ks, stride, pad)

        self.batch_norm = nn.BatchNorm2d(out_planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Run through convolution then BatchNorm then ReLU

        x = self.conv_upsample(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        return x
