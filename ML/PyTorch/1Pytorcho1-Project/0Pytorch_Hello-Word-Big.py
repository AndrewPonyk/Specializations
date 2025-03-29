import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Custom Dataset
class SimpleDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.x = torch.linspace(-10, 10, num_samples).unsqueeze(1).to(device)
        self.y = 2 * self.x + 3 + torch.randn(num_samples, 1).to(device) * 2

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# DataLoader
dataset = SimpleDataset()
batch_size = 32
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Model Definition
class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        out = self.linear(x)
        return out

model = SimpleLinearModel().to(device)
print(model)

# Loss and Optimizer
criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training Loop
num_epochs = 50
loss_values = []

for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(data_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    loss_values.append(loss.item())
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plot Loss Curve
plt.figure(figsize=(8, 4))
plt.plot(loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()

# Save and Load the Model
model_path = 'simple_linear_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

loaded_model = SimpleLinearModel().to(device)
loaded_model.load_state_dict(torch.load(model_path))
print("Model loaded and ready for inference.")

# Model Evaluation
loaded_model.eval()
with torch.no_grad():
    test_inputs = torch.linspace(-10, 10, 100).unsqueeze(1).to(device)
    test_outputs = loaded_model(test_inputs)

test_inputs = test_inputs.cpu().numpy()
test_outputs = test_outputs.cpu().numpy()

plt.figure(figsize=(8, 6))
plt.scatter(dataset.x.cpu().numpy(), dataset.y.cpu().numpy(), label='Original Data', alpha=0.5)
plt.plot(test_inputs, test_outputs, color='red', label='Fitted Line')
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.title('Linear Regression with PyTorch')
plt.legend()
plt.show()