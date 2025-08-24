import torch
import torchvision.models as models
import torch.nn as nn

def get_model(name="resnet18", num_classes=9, pretrained=True):
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
