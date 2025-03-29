import torch
import torch.nn as nn
from torchvision import models

# Load a pre-trained model
model = models.resnet18(pretrained=True)

# Freeze model weights
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer
model.fc = nn.Linear(model.fc.in_features, 10)  # Assuming 10 output classes

# Move the model to GPU if available
model = model.to(device)