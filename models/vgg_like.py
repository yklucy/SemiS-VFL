import torch
import torch.nn as nn
import torchvision.models as models

#####################################################################
# EMNIST
# Define a custom VGG model
class VGGLIKE(nn.Module):
    def __init__(self, input_shape=(1, 14, 28)):
        super(VGGLIKE, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 28, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(28, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(28, 28, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(28, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(28, 56, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(56, 56, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    

#####################################################################
# CIFAR10
# Define a custom VGG model
class VGGLIKE_CIFAR10(nn.Module):
    def __init__(self, input_shape=(3, 16, 32)):
        super(VGGLIKE_CIFAR10, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x