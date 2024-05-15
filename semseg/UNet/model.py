import math
import torch
import torch.nn as nn
import torch.nn.functional as NF
import torchvision.transforms.functional as F


class UNet(nn.Module):

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
            padding=0
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
            padding=0
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
            padding=0
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
            padding=0
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
            padding=0
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
            padding=0
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
            padding=0
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
            padding=0
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
            padding=0
        )

        self.class_conv = nn.Conv2d(
            in_channels=channels[0],
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.retain_size = retain_size

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

        concat_1 = torch.cat([bottleneck, F.center_crop(enc_pt_4, bottleneck.size()[-1])], dim=1)

        dec_pt_1 = self.decoder_pt_1(concat_1)
        dec_pt_1_uc = self.decoder_pt_1_uc(dec_pt_1)

        concat_2 = torch.cat([dec_pt_1_uc, F.center_crop(enc_pt_3, dec_pt_1_uc.size()[-1])], dim=1)

        dec_pt_2 = self.decoder_pt_2(concat_2)
        dec_pt_2_uc = self.decoder_pt_2_uc(dec_pt_2)

        concat_3 = torch.cat([dec_pt_2_uc, F.center_crop(enc_pt_2, dec_pt_2_uc.size()[-1])], dim=1)

        dec_pt_3 = self.decoder_pt_3(concat_3)
        dec_pt_3_uc = self.decoder_pt_3_uc(dec_pt_3)

        concat_4 = torch.cat([dec_pt_3_uc, F.center_crop(enc_pt_1, dec_pt_3_uc.size()[-1])], dim=1)

        dec_pt_4 = self.decoder_pt_4(concat_4)

        logits = self.class_conv(dec_pt_4)

        if self.retain_size:
            logits = NF.interpolate(
                input=logits, size=x.size()[-1], mode="bilinear", align_corners=False
            )

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
                kernel_size = m.kernel_size[0]
                num_features = m.out_channels

                mean = 0.0
                std = math.sqrt(2 / ((kernel_size ** 2) * num_features))

                nn.init.normal_(m.weight, mean=mean, std=std)
                nn.init.zeros_(m.bias)

        for m in self.conv_block_2.modules():
            if isinstance(m, nn.Conv2d):
                kernel_size = m.kernel_size[0]
                num_features = m.out_channels

                mean = 0.0
                std = math.sqrt(2 / ((kernel_size ** 2) * num_features))

                nn.init.normal_(m.weight, mean=mean, std=std)
                nn.init.zeros_(m.bias)

    def forward(self, x):

        conv_block_1_out = self.conv_block_1(x)
        conv_block_2_out = self.conv_block_2(conv_block_1_out)

        return conv_block_2_out
    