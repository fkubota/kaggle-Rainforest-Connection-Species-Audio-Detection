from ipdb import set_trace as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet18(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.__class__.__name__ = 'ResNet18'
        num_classes = params['n_classes']
        pretrained = params['pretrained']

        self.resnet = models.resnet18(pretrained=pretrained)
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


class ResNet18_2(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.__class__.__name__ = 'ResNet18'
        num_classes = params['n_classes']
        pretrained = params['pretrained']
        self.gap_ratio = params['gap_ratio']

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.resnet = models.resnet18(pretrained=pretrained)
        # in_features = self.resnet.fc.in_features
        # # self.resnet.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.resnet.avgpool = torch.cat([
        #         nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        #         nn.AdaptiveMaxPool2d(output_size=(1, 1))
        #         ])
        # self.resnet.fc = nn.Linear(in_features, num_classes)
        layers = list(self.resnet.children())[:-2]
        # layers.append(nn.AdaptiveMaxPool2d(1))
        self.encoder = nn.Sequential(*layers)

        in_features = self.resnet.fc.in_features
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        x = self.gap_ratio*self.avgpool(x) + (1-self.gap_ratio)*self.maxpool(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        multiclass_proba = F.softmax(x, dim=1)
        multilabel_proba = torch.sigmoid(x)
        return {
            "output": x,
            "output_softmax": multiclass_proba,
            "output_sigmoid": multilabel_proba
        }


class ResNet18_3(nn.Module):
    '''
    GAPとGMPのconcat
    '''
    def __init__(self, params):
        super().__init__()
        self.__class__.__name__ = 'ResNet18'
        num_classes = params['n_classes']
        pretrained = params['pretrained']
        self.gap_ratio = params['gap_ratio']

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.resnet = models.resnet18(pretrained=pretrained)
        layers = list(self.resnet.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        in_features = self.resnet.fc.in_features*2
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        x = torch.cat([
            self.gap_ratio*self.avgpool(x),
            (1 - self.gap_ratio)*self.maxpool(x)],
            axis=1)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        multiclass_proba = F.softmax(x, dim=1)
        multilabel_proba = torch.sigmoid(x)
        return {
            "output": x,
            "output_softmax": multiclass_proba,
            "output_sigmoid": multilabel_proba
        }
