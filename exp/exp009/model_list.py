from ipdb import set_trace as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


####################################################################
#     Resnet50
####################################################################
class ResNet18(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.__class__.__name__ = 'ResNet18'
        num_classes = params['n_classes']
        pretrained = params['pretrained']

        self.resnet = models.resnet18(pretrained=pretrained)
        in_features = self.resnet.fc.in_features
        # self.resnet.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.resnet.avgpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        # self.resnet.avgpool = nn.AdaptiveMaxPool2d(output_size=(1, 1)) +\
            # nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        multiclass_proba = F.softmax(x, dim=1)
        multilabel_proba = torch.sigmoid(x)
        return {
            "output": x,
            "output_softmax": multiclass_proba,
            "output_sigmoid": multilabel_proba
        }


class ResNet50(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.__class__.__name__ = 'ResNet50'
        num_classes = params['n_classes']
        pretrained = params['pretrained']

        self.resnet = models.resnet50(pretrained=pretrained)
        in_features = self.resnet.fc.in_features
        self.resnet.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        multiclass_proba = F.softmax(x, dim=1)
        multilabel_proba = torch.sigmoid(x)
        return {
            "output": x,
            "output_softmax": multiclass_proba,
            "output_sigmoid": multilabel_proba
        }
