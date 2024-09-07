import torch
import torch.nn as nn
import torch.nn.functional as NF
import torchvision.transforms.functional as F

from cv.utils import MetaWrapper

class UNet(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Model Class for UNet architecture from paper on: Convolutional Networks for Biomedical Image Segmentation"

    def __init__(
            self,
            channels=[64, 128, 256, 512, 1024],
            in_channels=3,
            num_classes=81,
            dropout=0.5,
            retain_size=False
        ):

        super(UNet, self).__init__()

        self.encoder_part_1 = ConvBlock(
            in_channels=in_channels,
            out_channels=channels[0],
            kernel_size=3,
            stride=1,
            padding=0 if not retain_size else 1
        )
        self.encoder_part_1_max_pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.encoder_part_2 = ConvBlock(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=3,
            stride=1,
            padding=0 if not retain_size else 1
        )
        self.encoder_part_2_max_pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.encoder_part_3 = ConvBlock(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            padding=0 if not retain_size else 1
        )
        self.encoder_part_3_max_pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.encoder_part_4 = ConvBlock(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=3,
            stride=1,
            padding=0 if not retain_size else 1
        )
        self.encoder_part_4_max_pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.encoder_part_5 = ConvBlock(
            in_channels=channels[3],
            out_channels=channels[4],
            kernel_size=3,
            stride=1,
            padding=0 if not retain_size else 1
        )

        self.dropout = nn.Dropout(p=dropout)

        self.bottleneck = nn.ConvTranspose2d(
            in_channels=channels[4],
            out_channels=channels[3],
            kernel_size=2,
            stride=2,
            padding=0
        )

        self.decoder_pt_1 = ConvBlock(
            in_channels=channels[4],
            out_channels=channels[3],
            kernel_size=3,
            stride=1,
            padding=0 if not retain_size else 1
        )
        self.decoder_pt_1_uc = nn.ConvTranspose2d(
            in_channels=channels[3],
            out_channels=channels[2],
            kernel_size=2,
            stride=2,
            padding=0
        )

        self.decoder_pt_2 = ConvBlock(
            in_channels=channels[3],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            padding=0 if not retain_size else 1
        )
        self.decoder_pt_2_uc = nn.ConvTranspose2d(
            in_channels=channels[2],
            out_channels=channels[1],
            kernel_size=2,
            stride=2,
            padding=0
        )

        self.decoder_pt_3 = ConvBlock(
            in_channels=channels[2],
            out_channels=channels[1],
            kernel_size=3,
            stride=1,
            padding=0 if not retain_size else 1
        )
        self.decoder_pt_3_uc = nn.ConvTranspose2d(
            in_channels=channels[1],
            out_channels=channels[0],
            kernel_size=2,
            stride=2,
            padding=0
        )

        self.decoder_pt_4 = ConvBlock(
            in_channels=channels[1],
            out_channels=channels[0],
            kernel_size=3,
            stride=1,
            padding=0 if not retain_size else 1
        )

        self.class_conv = nn.Conv2d(
            in_channels=channels[0],
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def _crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = NF.pad(bypass, (-c, -c, -c, -c))

        return torch.cat((upsampled, bypass), dim=1)

    def forward(self, x):
        enc_pt_1 = self.encoder_part_1(x)
        enc_pt_1_mp = self.encoder_part_1_max_pool(enc_pt_1)

        enc_pt_2 = self.encoder_part_2(enc_pt_1_mp)
        enc_pt_2_mp = self.encoder_part_2_max_pool(enc_pt_2)

        enc_pt_3 = self.encoder_part_3(enc_pt_2_mp)
        enc_pt_3_mp = self.encoder_part_3_max_pool(enc_pt_3)

        enc_pt_4 = self.encoder_part_4(enc_pt_3_mp)
        enc_pt_4_mp = self.encoder_part_4_max_pool(enc_pt_4)

        enc_pt_5 = self.encoder_part_5(enc_pt_4_mp)

        bottleneck = self.bottleneck(enc_pt_5)

        concat_1 = self._crop_and_concat(upsampled=bottleneck, bypass=enc_pt_4, crop=True)

        dec_pt_1 = self.decoder_pt_1(concat_1)
        dec_pt_1_uc = self.decoder_pt_1_uc(dec_pt_1)

        concat_2 = self._crop_and_concat(upsampled=dec_pt_1_uc, bypass=enc_pt_3, crop=True)

        dec_pt_2 = self.decoder_pt_2(concat_2)
        dec_pt_2_uc = self.decoder_pt_2_uc(dec_pt_2)

        concat_3 = self._crop_and_concat(upsampled=dec_pt_2_uc, bypass=enc_pt_2, crop=True)

        dec_pt_3 = self.decoder_pt_3(concat_3)
        dec_pt_3_uc = self.decoder_pt_3_uc(dec_pt_3)

        concat_4 = self._crop_and_concat(upsampled=dec_pt_3_uc, bypass=enc_pt_1, crop=True)

        dec_pt_4 = self.decoder_pt_4(concat_4)

        logits = self.class_conv(dec_pt_4)

        return logits


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):

        super(ConvBlock, self).__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True))
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

        self.initialize_conv()

    def initialize_conv(self):

        for m in self.conv_block_1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

        for m in self.conv_block_2.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):

        conv_block_1_out = self.conv_block_1(x)
        conv_block_2_out = self.conv_block_2(conv_block_1_out)

        return conv_block_2_out
