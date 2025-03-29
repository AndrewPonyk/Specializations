import torch

# Create a tensor from a list
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print("Tensor x:\n", x)

# Create a random tensor
y = torch.rand((2, 2))
print("Random Tensor y:\n", y)

# Tensor addition
z = x + y
print("Sum of x and y:\n", z)

# Element-wise multiplication
w = x * y
print("Element-wise multiplication of x and y:\n", w)

# Matrix multiplication
m = torch.matmul(x, y)
print("Matrix multiplication of x and y:\n", m)