import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, outputs, targets):
        loss = torch.mean((outputs - targets) ** 2)
        return loss

# Usage
criterion = CustomLoss()