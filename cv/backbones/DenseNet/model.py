import math
import torch
import numpy as np
import torch.nn as nn

from cv.utils import MetaWrapper

"""
The code has nested classes, which I later realised is not a good practice
"""

class DenseNet(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Model Class for DenseNet architecture from paper on: Densely Connected Convolutional Networks"

    class DenseLayer(nn.Module):
        def __init__(self, in_features, k=32, dropout_rate=None):
            super(DenseNet.DenseLayer, self).__init__()

            self.dense_layer = nn.Sequential(
                DenseNet.CompositeFunction(
                    in_channels=in_features, out_channels=4*k,
                    kernel_size=1, stride=1, padding=0
                ),
                DenseNet.CompositeFunction(
                    in_channels=4*k, out_channels=k,
                    kernel_size=3, stride=1, padding=1
                )
            )

            if dropout_rate is not None:
                self.dropout = nn.Dropout(p=dropout_rate)
            else:
                self.dropout = None

        def forward(self, x):
            out = self.dense_layer(x)
            if self.dropout is not None:
                out = self.dropout(out)

            return out

    class DenseBlock(nn.ModuleDict):
        def __init__(self, num_layers, in_features, k, dropout_rate=None):
            super(DenseNet.DenseBlock, self).__init__()

            for i in range(num_layers):
                layer = DenseNet.DenseLayer(
                    in_features=in_features, k=k, dropout_rate=dropout_rate
                )
                self.add_module(f"dense_layer_{i+1}", layer)
                in_features += k

        def forward(self, x):
            next_layer_in = [x]
            for dense_layer in self.values():
                layer_in = torch.cat(next_layer_in, dim=1)
                layer_out = dense_layer(layer_in)
                next_layer_in.append(layer_out)

            return torch.cat(next_layer_in, dim=1)

    class CompositeFunction(nn.Module):
        def __init__(self, in_channels, out_channels,
                     kernel_size, stride, padding):
            super(DenseNet.CompositeFunction, self).__init__()

            self.bn = nn.BatchNorm2d(num_features=in_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding
            )

            self.initializeConv(kernel_size=kernel_size, channels=out_channels)

        def initializeConv(self, kernel_size, channels):
            init_n = (kernel_size ** 2) * channels
            init_mean = 0.0
            init_std = np.sqrt(2 / init_n)
            
            nn.init.normal_(self.conv.weight, mean=init_mean, std=init_std)
            nn.init.constant_(self.conv.bias, val=0.0)

        def forward(self, x):
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv(x)

            return x
        
    class TransitionLayer(nn.Module):
        def __init__(self, in_features, theta=0.5):

            super(DenseNet.TransitionLayer, self).__init__()

            out_features = math.floor(theta * in_features)

            self.transition_layer = nn.Sequential(
                DenseNet.CompositeFunction(
                    in_channels=in_features, out_channels=out_features,
                    kernel_size=1, stride=1, padding="same"
                ),
                nn.AvgPool2d(kernel_size=2, stride=2)
            )

        def forward(self, x):
            return self.transition_layer(x)

    def __init__(self, num_classes, dense_block_num_layers=[6, 12, 24, 16],
                 in_channels=3, growth_rate=32, compression_factor=0.5, dense_block_dropouts=None):
        
        super(DenseNet, self).__init__()

        self.init_features = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=2*growth_rate,
                kernel_size=7, stride=2, padding=3
            ),
            nn.BatchNorm2d(num_features=2*growth_rate),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        dense_block_1_in_features = 2*growth_rate
        self.dense_block_1 = DenseNet.DenseBlock(
            num_layers=dense_block_num_layers[0], in_features=dense_block_1_in_features,
            k=growth_rate, dropout_rate=dense_block_dropouts
        )

        transition_layer_1_in_features = 2*growth_rate+dense_block_num_layers[0]*growth_rate
        self.transition_layer_1 = DenseNet.TransitionLayer(
            in_features=transition_layer_1_in_features, theta=compression_factor
        )

        dense_block_2_in_features = math.floor(compression_factor*transition_layer_1_in_features)
        self.dense_block_2 = DenseNet.DenseBlock(
            num_layers=dense_block_num_layers[1], in_features=dense_block_2_in_features,
            k=growth_rate, dropout_rate=dense_block_dropouts
        )

        transition_layer_2_in_features = dense_block_2_in_features+dense_block_num_layers[1]*growth_rate
        self.transition_layer_2 = DenseNet.TransitionLayer(
            in_features=transition_layer_2_in_features, theta=compression_factor
        )

        dense_block_3_in_features = math.floor(compression_factor*transition_layer_2_in_features)
        self.dense_block_3 = DenseNet.DenseBlock(
            num_layers=dense_block_num_layers[2], in_features=dense_block_3_in_features,
            k=growth_rate, dropout_rate=dense_block_dropouts
        )

        transition_layer_3_in_features = dense_block_3_in_features+dense_block_num_layers[2]*growth_rate
        self.transition_layer_3 = DenseNet.TransitionLayer(
            in_features=transition_layer_3_in_features, theta=compression_factor
        )

        dense_block_4_in_features = math.floor(compression_factor*transition_layer_3_in_features)
        self.dense_block_4 = DenseNet.DenseBlock(
            num_layers=dense_block_num_layers[3], in_features=dense_block_4_in_features,
            k=growth_rate, dropout_rate=dense_block_dropouts
        )

        classifier_in_features = dense_block_4_in_features+dense_block_num_layers[3]*growth_rate
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=classifier_in_features, out_features=num_classes)
        )

        self.initializeConv()

    def initializeConv(self):
        for module in self.init_features.modules():
            if isinstance(module, nn.Conv2d):
                weight = module.weight
                bias = module.bias

                init_n = (module.kernel_size[0] ** 2) * module.out_channels
                init_mean = 0.0
                init_std = np.sqrt(2 / init_n)

                nn.init.normal_(weight, mean=init_mean, std=init_std)
                nn.init.constant_(bias, val=0.0)
        
    def forward(self, x):

        x = self.init_features(x)
        x = self.dense_block_1(x)
        x = self.transition_layer_1(x)
        x = self.dense_block_2(x)
        x = self.transition_layer_2(x)
        x = self.dense_block_3(x)
        x = self.transition_layer_3(x)
        x = self.dense_block_4(x)
        x = self.classifier(x)

        return x
