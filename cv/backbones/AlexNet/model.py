import torch
import torch.nn as nn
from collections import OrderedDict

from cv.utils import MetaWrapper

class AlexNet(nn.Module, metaclass=MetaWrapper):

    """
    AlexNet model for image classification based on the `paper <https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf>`_.

    This class defines the AlexNet architecture, which consists of convolutional layers for feature extraction 
    and fully connected layers for classification.

    Methods:
        __init__(self, num_classes, in_channels=3):
            Initializes the AlexNet model with the number of classes for classification and input channels.
        forward(self, x):
            Defines the forward pass of the AlexNet model, applying feature extraction followed by classification.

    Example:
        >>> model = AlexNet(num_classes=1000, in_channels=3)
        >>> output = model(torch.randn(1, 3, 224, 224))  # Example input tensor of shape (batch_size, channels, height, width)
    """


    @classmethod
    def __class_repr__(cls):
        return "Model Class for AlexNet architecture from paper on: ImageNet Classification with Deep Convolutional Neural Networks"
    
    def __init__(self, num_classes, in_channels=3):

        """
        Initializes the AlexNet model.

        Args:
            num_classes (int): The number of output classes for classification.
            in_channels (int, optional): The number of input channels for the images. Default is 3 for RGB images.

        Example:
            >>> model = AlexNet(num_classes=1000, in_channels=3)

        """
        
        super(AlexNet, self).__init__()

        self.feature_extractor_layers = OrderedDict(
            [
                ("conv1", nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=11, stride=4)),
                ("relu1", nn.ReLU(inplace=True)),
                ("pool1", nn.MaxPool2d(kernel_size=3, stride=2)),

                ("conv2", nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)),
                ("relu2", nn.ReLU(inplace=True)),
                ("pool2", nn.MaxPool2d(kernel_size=3, stride=2)),

                ("conv3", nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)),
                ("relu3", nn.ReLU(inplace=True)),

                ("conv4", nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)),
                ("relu4", nn.ReLU(inplace=True)),

                ("conv5", nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)),
                ("relu5", nn.ReLU(inplace=True)),
                ("pool5", nn.MaxPool2d(kernel_size=3, stride=2)),
            ]
        )

        self.feature_extractor = nn.Sequential(self.feature_extractor_layers)

        self.classifier_layers = OrderedDict(
            [
                ("dropout1", nn.Dropout(p=0.5)),
                ("fc1", nn.Linear(256*5*5, 4096)),
                ("relu_fc1", nn.ReLU(inplace=True)),
                ("dropout2", nn.Dropout(p=0.5)),
                ("fc2", nn.Linear(4096, 4096)),
                ("relu_fc2", nn.ReLU(inplace=True)),
                ("fc_class", nn.Linear(4096, num_classes))
            ]
        )

        self.classifier = nn.Sequential(self.classifier_layers)

    def forward(self, x):

        """
        Defines the forward pass of the AlexNet model.

        The input tensor passes through the feature extractor (convolutional layers), 
        then is flattened and passed through the classifier (fully connected layers).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor after passing through the model, with shape (batch_size, num_classes).

        Example:
            >>> output = model(torch.randn(1, 3, 224, 224))

        """
            
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)

        return x
