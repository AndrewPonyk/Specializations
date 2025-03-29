import torch
import torch.nn as nn

# Define a model
model = nn.Linear(10, 1)

# Train the model (omitted for brevity)

# Save the model's state_dict
torch.save(model.state_dict(), 'model.pth')
print("Model saved to model.pth")

# To load the model later
model_loaded = nn.Linear(10, 1)
model_loaded.load_state_dict(torch.load('model.pth'))
model_loaded.eval()  # Set the model to evaluation mode

# Verify that the model parameters are the same
for param1, param2 in zip(model.parameters(), model_loaded.parameters()):
    print(torch.equal(param1, param2))  # Should print True