import torch

# Enable gradient computation on tensor
x = torch.tensor(2.0, requires_grad=True)

# Define a function
y = x ** 3 + 2 * x

# Compute the gradient
y.backward()

# Print the gradient dy/dx at x=2.0
print("Gradient dy/dx at x=2.0:", x.grad.item())