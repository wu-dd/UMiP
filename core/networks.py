import torch
import torch.nn as nn
import torch.nn.functional as F
from lib import resnet


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.n_outputs = 84

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, (2, 2))
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, (2, 2))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return out


def Featurizer(input_shape, hparams):
    """Backbone selected by hparams['model']; exposes .n_outputs."""
    if hparams["model"] == "LeNet":
        return LeNet()
    elif hparams["model"] == "ResNet32":
        return resnet.resnet(depth=32)
    elif hparams["model"] == "ResNet34":
        from core.resnet_dirk import resnet34
        return resnet34()
    elif hparams["model"] == 'ResNet110':
        return resnet.resnet(depth=110)
    else:
        raise NotImplementedError


def Classifier(in_features, out_features):
    return torch.nn.Linear(in_features, out_features)
