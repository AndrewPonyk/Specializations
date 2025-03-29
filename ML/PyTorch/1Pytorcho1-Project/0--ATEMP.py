"""
PyTorch RNN Implementation for Sequence Prediction
This script demonstrates a simple Recurrent Neural Network (RNN) implementation 
using PyTorch that learns to predict the next number in a sequence where each 
number is the sum of the previous two numbers (similar to Fibonacci sequence but with random starting values).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Define the RNN model
class SimpleRNN(nn.Module):
    """
    A simple single-layer RNN model for sequence prediction.
    
    Args:
        input_size (int): Size of each input feature
        hidden_size (int): Number of features in the hidden state
        output_size (int): Size of output per sequence
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        # Create a single-layer RNN with specified dimensions
        # batch_first=True means input shape is (batch_size, seq_len, input_size)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # Fully connected layer to transform hidden state to output
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass of the RNN.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Tensor: Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden state with zeros (1 is for num_layers, x.size(0) is batch_size)
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        
        # Forward propagate RNN: output shape (batch_size, seq_len, hidden_size)
        # We don't need the final hidden state, so we use _ to ignore it
        out, _ = self.rnn(x, h0)
        
        # Extract the output from the last time step only
        # Shape becomes (batch_size, hidden_size)
        last_time_step = out[:, -1, :]
        
        # Pass through the fully connected layer to get final output
        # Shape becomes (batch_size, output_size)
        out = self.fc(last_time_step)
        return out

# Generate synthetic data for training and testing
def generate_sequence(n_samples):
    """
    Generate sequences where each number is the sum of the previous two numbers.
    
    Args:
        n_samples (int): Number of sequence samples to generate
        
    Returns:
        tuple: (sequences, targets) where:
            - sequences is a tensor of shape (n_samples, 4, 1) containing input sequences
            - targets is a tensor of shape (n_samples,) containing target values
    """
    sequences = []
    targets = []
    for _ in range(n_samples):
        # Start with two random numbers between 0 and 1
        seq = [np.random.rand(), np.random.rand()]
        
        # Generate 3 more numbers in the sequence (each is sum of previous two)
        for _ in range(3):
            seq.append(seq[-1] + seq[-2])
        
        # Use first 4 numbers as input sequence
        sequences.append(seq[:-1])
        # Use the 5th number as the target to predict
        targets.append(seq[-1])
    
    # Convert to PyTorch tensors
    return torch.FloatTensor(sequences), torch.FloatTensor(targets)

# Create training dataset with 100 samples
X_train, y_train = generate_sequence(100)
# Add input dimension to make shape (batch_size, seq_len, input_size)
# This transforms from shape (100, 4) to (100, 4, 1)
X_train = X_train.unsqueeze(-1)  

# Model configuration
input_size = 1       # Each time step has a single feature
hidden_size = 32     # Size of the hidden state
output_size = 1      # Predicting a single value

# Initialize model, loss function, and optimizer
model = SimpleRNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
# Mean Squared Error loss is appropriate for regression problems
criterion = nn.MSELoss()
# Adam optimizer with learning rate of 0.01
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop configuration
n_epochs = 200       # Number of complete passes through the training dataset
losses = []          # To store loss values for plotting

# Training loop
for epoch in range(n_epochs):
    # Reset gradients to zero before each backpropagation
    model.zero_grad()
    
    # Forward pass: compute model predictions
    output = model(X_train)
    
    # Compute loss: compare predictions with actual targets
    # Squeeze output to match target shape
    loss = criterion(output.squeeze(), y_train)
    
    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()
    
    # Update model parameters based on gradients
    optimizer.step()
    
    # Store loss for plotting
    losses.append(loss.item())
    
    # Print progress every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

# Visualize training progress
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.grid(True)
plt.show()

# Evaluate model performance on new data
print("\n--- Model Evaluation ---")
model.eval()  # Set model to evaluation mode (disables dropout, etc.)
with torch.no_grad():  # Disable gradient calculation for inference
    # Generate a single test sequence
    test_seq, test_target = generate_sequence(1)
    test_seq = test_seq.unsqueeze(-1)  # Add input dimension
    
    # Make prediction
    prediction = model(test_seq)
    
    # Display results
    print(f'Test Sequence: {test_seq.squeeze().tolist()}')
    print(f'Predicted Next Value: {prediction.item():.4f}')
    print(f'Actual Next Value: {test_target.item():.4f}')
    print(f'Prediction Error: {abs(prediction.item() - test_target.item()):.4f}')