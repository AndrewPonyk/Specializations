import torch
import torch.nn as nn

print("Cuda is available:" + str(torch.cuda.is_available()))
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Define a simple model and move it to the device
model = nn.Linear(10, 1).to(device)

# Dummy input and move it to the device
x = torch.randn(5, 10).to(device)
output = model(x)
print("Output:", output)