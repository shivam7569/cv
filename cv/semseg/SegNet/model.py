import torch.nn as nn

from cv.src.checkpoints import Checkpoint
from cv.utils import MetaWrapper

class SegNet(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Model Class for SegNet architecture from paper on: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation"

    def __init__(self, num_classes, in_channels=3):

        super(SegNet, self).__init__()

        self.enc_conv_1 = ConvBlock(
            in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.enc_conv_2 = ConvBlock(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.enc_maxpool_1 = nn.MaxPool2d(
            kernel_size=2, stride=2, return_indices=True
        )
        self.enc_conv_3 = ConvBlock(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.enc_conv_4 = ConvBlock(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.enc_maxpool_2 = nn.MaxPool2d(
            kernel_size=2, stride=2, return_indices=True
        )
        self.enc_conv_5 = ConvBlock(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.enc_conv_6 = ConvBlock(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.enc_conv_7 = ConvBlock(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.enc_maxpool_3 = nn.MaxPool2d(
            kernel_size=2, stride=2, return_indices=True
        )
        self.enc_conv_8 = ConvBlock(
            in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.enc_conv_9 = ConvBlock(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.enc_conv_10 = ConvBlock(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.enc_maxpool_4 = nn.MaxPool2d(
            kernel_size=2, stride=2, return_indices=True
        )
        self.enc_conv_11 = ConvBlock(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.enc_conv_12 = ConvBlock(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.enc_conv_13 = ConvBlock(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.enc_maxpool_5 = nn.MaxPool2d(
            kernel_size=2, stride=2, return_indices=True
        )

        self.dec_unpool_1 = nn.MaxUnpool2d(
            kernel_size=2, stride=2
        )
        self.dec_conv_1 = ConvBlock(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.dec_conv_2 = ConvBlock(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.dec_conv_3 = ConvBlock(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.dec_unpool_2 = nn.MaxUnpool2d(
            kernel_size=2, stride=2
        )
        self.dec_conv_4 = ConvBlock(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.dec_conv_5 = ConvBlock(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.dec_conv_6 = ConvBlock(
            in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.dim_reduction_1 = ConvBlock(
            in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0
        )
        self.dec_unpool_3 = nn.MaxUnpool2d(
            kernel_size=2, stride=2
        )
        self.dec_conv_7 = ConvBlock(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.dec_conv_8 = ConvBlock(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.dec_conv_9 = ConvBlock(
            in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.dim_reduction_2 = ConvBlock(
            in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0
        )
        self.dec_unpool_4 = nn.MaxUnpool2d(
            kernel_size=2, stride=2
        )
        self.dec_conv_10 = ConvBlock(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.dec_conv_11 = ConvBlock(
            in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.dim_reduction_3 = ConvBlock(
            in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0
        )
        self.dec_unpool_5 = nn.MaxUnpool2d(
            kernel_size=2, stride=2
        )
        self.dec_conv_12 = ConvBlock(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.dec_conv_13 = ConvBlock(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )

        self.classifier = nn.Conv2d(
            in_channels=64, out_channels=num_classes, kernel_size=1, stride=1, padding=0
        )

        self.initialize_weights()

    def initialize_weights(self):
        checkpoint = Checkpoint.load(None, name="VGG16", return_checkpoint=True)
        weights = {k.replace("feature_extractor.", ''): v for k, v in checkpoint["model_state_dict"].items() if k.split('.')[0] == "feature_extractor"}
        for name, param in weights.items():
            layer_name, param_name = name.split('.')
            layer_name = "enc_" + layer_name[:4] + "_" + layer_name[4:]
            convLayer = getattr(self, layer_name).convBlock[0]
            if param_name == "weight":
                convLayer.weight = nn.Parameter(param)
            if param_name == "bias":
                convLayer.bias = nn.Parameter(param)

    def forward(self, x):
        enc_conv1 = self.enc_conv_1(x)
        enc_conv2 = self.enc_conv_2(enc_conv1)
        enc_maxpool1, enc_indices1 = self.enc_maxpool_1(enc_conv2)
        enc_conv3 = self.enc_conv_3(enc_maxpool1)
        enc_conv4 = self.enc_conv_4(enc_conv3)
        enc_maxpool2, enc_indices2 = self.enc_maxpool_2(enc_conv4)
        enc_conv5 = self.enc_conv_5(enc_maxpool2)
        enc_conv6 = self.enc_conv_6(enc_conv5)
        enc_conv7 = self.enc_conv_7(enc_conv6)
        enc_maxpool3, enc_indices3 = self.enc_maxpool_3(enc_conv7)
        enc_conv8 = self.enc_conv_8(enc_maxpool3)
        enc_conv9 = self.enc_conv_9(enc_conv8)
        enc_conv10 = self.enc_conv_10(enc_conv9)
        enc_maxpool4, enc_indices4 = self.enc_maxpool_4(enc_conv10)
        enc_conv11 = self.enc_conv_11(enc_maxpool4)
        enc_conv12 = self.enc_conv_12(enc_conv11)
        enc_conv13 = self.enc_conv_13(enc_conv12)
        enc_maxpool5, enc_indices5 = self.enc_maxpool_5(enc_conv13)

        dec_unpool1 = self.dec_unpool_1(enc_maxpool5, enc_indices5)
        dec_conv1 = self.dec_conv_1(dec_unpool1) + enc_conv13
        dec_conv2 = self.dec_conv_2(dec_conv1) + enc_conv12
        dec_conv3 = self.dec_conv_3(dec_conv2) + enc_conv11
        dec_unpool2 = self.dec_unpool_2(dec_conv3, enc_indices4)
        dec_conv4 = self.dec_conv_4(dec_unpool2) + enc_conv10
        dec_conv5 = self.dec_conv_5(dec_conv4) + enc_conv9
        dec_conv6 = self.dec_conv_6(dec_conv5) + self.dim_reduction_1(enc_conv8)
        dec_unpool3 = self.dec_unpool_3(dec_conv6, enc_indices3)
        dec_conv7 = self.dec_conv_7(dec_unpool3) + enc_conv7
        dec_conv8 = self.dec_conv_8(dec_conv7) + enc_conv6
        dec_conv9 = self.dec_conv_9(dec_conv8) + self.dim_reduction_2(enc_conv5)
        dec_unpool4 = self.dec_unpool_4(dec_conv9, enc_indices2)
        dec_conv10 = self.dec_conv_10(dec_unpool4) + enc_conv4
        dec_conv11 = self.dec_conv_11(dec_conv10) + self.dim_reduction_3(enc_conv3)
        dec_unpool5 = self.dec_unpool_5(dec_conv11, enc_indices1)
        dec_conv12 = self.dec_conv_12(dec_unpool5) + enc_conv2
        dec_conv13 = self.dec_conv_13(dec_conv12) + enc_conv1

        logits = self.classifier(dec_conv13)

        return logits


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):

        super(ConvBlock, self).__init__()

        self.convBlock = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

        self.initialize()

    def initialize(self):
        convLayer = self.convBlock[0]
        nn.init.kaiming_normal_(convLayer.weight)
        nn.init.constant_(convLayer.bias, val=0.0)

    def forward(self, x):
        x = self.convBlock(x)

        return x
