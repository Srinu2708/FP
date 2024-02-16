import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def last_block(in_channels, mid_channels, out_channels, kernel_size=3):
    # If a nn.Conv layer is directly followed by a nn.BatchNorm layer, then the bias in the convolution is not needed
    block = nn.Sequential(
        nn.Conv3d(in_channels=in_channels + mid_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=1,
                  padding=1, bias=False),
        nn.BatchNorm3d(num_features=mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=1, padding=1,
                  bias=False),
        nn.BatchNorm3d(num_features=mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1,
                  bias=False),
        nn.BatchNorm3d(num_features=out_channels),
        nn.Softmax(dim=1)
    )
    return block


def crop_and_concat(bypass, upsampled, crop=False):
    if crop:
        z_diff = bypass.shape[2] - upsampled.shape[2]
        y_diff = bypass.shape[3] - upsampled.shape[3]
        x_diff = bypass.shape[4] - upsampled.shape[4]
        upsampled = F.pad(upsampled,
                          (math.floor(x_diff / 2), math.ceil(x_diff / 2),
                           math.floor(y_diff / 2), math.ceil(y_diff / 2),

                           math.floor(z_diff / 2), math.ceil(z_diff / 2)))
    return torch.cat([bypass, upsampled], dim=1)


def upsample_block(in_channels, out_channels, kernel_size=3):
    # If a nn.Conv layer is directly followed by a nn.BatchNorm layer, then the bias in the convolution is not needed
    block = nn.Sequential(
        nn.Conv3d(in_channels=in_channels + out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                  padding=1, bias=False),
        nn.BatchNorm3d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1,
                  bias=False),
        nn.BatchNorm3d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
    )
    return block


class UNet3D(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(UNet3D, self).__init__()

        self.conv_encode1 = conv_block(in_channels=in_channel, out_channels=32)
        self.conv_maxpool1 = nn.MaxPool3d(kernel_size=2)
        self.conv_encode2 = conv_block(in_channels=32, out_channels=64)
        self.conv_maxpool2 = nn.MaxPool3d(kernel_size=2)
        self.conv_encode3 = conv_block(in_channels=64, out_channels=128)
        self.conv_maxpool3 = nn.MaxPool3d(kernel_size=2)
        self.conv_encode4 = conv_block(in_channels=128, out_channels=256)
        self.conv_maxpool4 = nn.MaxPool3d(kernel_size=2)
        # If a nn.Conv layer is directly followed by a nn.BatchNorm layer, then the bias in the convolution is not needed
        self.bottleneck = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(in_channels=512, out_channels=512, kernel_size=2, stride=2, bias=True)
        )

        self.conv_decode4 = upsample_block(in_channels=512, out_channels=256)
        self.conv_decode3 = upsample_block(in_channels=256, out_channels=128)
        self.conv_decode2 = upsample_block(in_channels=128, out_channels=64)
        self.final_layer = last_block(in_channels=64, mid_channels=32, out_channels=out_channel)

    def forward(self, x):
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        encode_block4 = self.conv_encode4(encode_pool3)
        encode_pool4 = self.conv_maxpool4(encode_block4)

        bottleneck = self.bottleneck(encode_pool4)

        decode_block4 = crop_and_concat(encode_block4, bottleneck, crop=True)
        cat_layer3 = self.conv_decode4(decode_block4)
        decode_block3 = crop_and_concat(encode_block3, cat_layer3, crop=True)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = crop_and_concat(encode_block2, cat_layer2, crop=True)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = crop_and_concat(encode_block1, cat_layer1, crop=True)

        final_layer = self.final_layer(decode_block1)

        return final_layer


def conv_block(in_channels, out_channels, kernel_size=3):
    # If a nn.Conv layer is directly followed by a nn.BatchNorm layer, then the bias in the convolution is not needed
    block = nn.Sequential(
        nn.Conv3d(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=kernel_size, stride=1, padding=1,
                  bias=False),
        nn.BatchNorm3d(num_features=out_channels // 2),
        nn.ReLU(inplace=True),
        nn.Conv3d(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                  padding=1, bias=False),
        nn.BatchNorm3d(num_features=out_channels),
        nn.ReLU(inplace=True)
    )
    return block
