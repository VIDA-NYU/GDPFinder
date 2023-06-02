import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def generate_resnet(device):

    # Define ResNet-50 model
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features

    # Freeze the base model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Add additional fully connected layers
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, 32),
        nn.ReLU(inplace=True),
        nn.Linear(32, 1)
    )
    model.to(device)

    # Define loss function
    criterion = nn.L1Loss()
    
    return model, criterion