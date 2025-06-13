import numpy as np
import matplotlib.pyplot as plt

# Simple Neural Network for Diabetes Prediction
# This example uses the Pima Indians Diabetes dataset from the UCI Machine Learning Repository
# The goal is to predict whether a patient has diabetes based on simple health measurements

# ----- PART 1: LOADING REAL DATA -----
print("Loading diabetes dataset...")

# Real dataset: Pima Indians Diabetes Dataset
# Features:
# 1. Number of pregnancies
# 2. Glucose level
# 3. Blood pressure
# 4. Skin thickness
# 5. Insulin level
# 6. BMI
# 7. Diabetes pedigree function
# 8. Age
# Target: Has diabetes (1) or not (0)

# Load a small subset of the real data for simplicity
# Format: [pregnancies, glucose, blood_pressure, bmi, diabetes_pedigree, age, outcome]
# We'll use fewer features for simplicity (glucose and bmi are most predictive)
diabetes_data = np.array([
    # [glucose, bmi, outcome]
    [89, 26.4, 0],  # Healthy
    [137, 40.6, 1],  # Diabetic
    [78, 31.2, 0],  # Healthy
    [197, 34.7, 1],  # Diabetic
    [119, 35.5, 0],  # Healthy
    [167, 33.6, 1],  # Diabetic
    [118, 28.2, 0],  # Healthy
    [110, 24.3, 0],  # Healthy
    [168, 38.2, 1],  # Diabetic
    [139, 43.1, 1],  # Diabetic
    [130, 23.1, 0],  # Healthy
    [85, 26.6, 0],   # Healthy
    [176, 44.5, 1],  # Diabetic
    [154, 27.8, 1],  # Diabetic
    [104, 30.8, 0],  # Healthy
    [95, 28.7, 0]    # Healthy
])

# Separate features and labels
X = diabetes_data[:, 0:2]  # Features: glucose and BMI
y = diabetes_data[:, 2]    # Labels: diabetic or not

# Normalize data - important for neural networks!
# Scale glucose to range [0, 1]
X[:, 0] = (X[:, 0] - 70) / (200 - 70)
# Scale BMI to range [0, 1]
X[:, 1] = (X[:, 1] - 20) / (50 - 20)

# ----- PART 2: NEURAL NETWORK FUNCTIONS -----

# Activation function: Sigmoid turns any value into a number between 0 and 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid for backpropagation
def sigmoid_derivative(x):
    return x * (1 - x)

# Forward pass - how the neural network makes predictions
def forward(inputs, weights1, weights2):
    # First layer calculation (inputs -> hidden layer)
    layer1 = sigmoid(np.dot(inputs, weights1))
    # Output layer calculation (hidden layer -> output)
    output = sigmoid(np.dot(layer1, weights2))
    return layer1, output

# ----- PART 3: TRAINING THE NEURAL NETWORK -----

# Set hyperparameters
input_size = 2        # Two features: glucose and BMI
hidden_size = 4       # Small hidden layer for this simple example
output_size = 1       # Binary output: diabetic or not
learning_rate = 0.5   # How quickly the network learns
epochs = 1000         # Training iterations

# Initialize weights with random values
np.random.seed(42)  # For reproducibility
weights1 = np.random.uniform(size=(input_size, hidden_size))
weights2 = np.random.uniform(size=(hidden_size, output_size))

# Keep track of error over time
error_history = []

# Training loop
print("\nTraining the neural network...")
for epoch in range(epochs):
    # Forward pass (prediction)
    layer1, output = forward(X, weights1, weights2)
    
    # Calculate error
    error = y.reshape(-1, 1) - output
    mean_error = np.abs(error).mean()
    error_history.append(mean_error)
    
    # Backward pass (learning)
    # Calculate how much to adjust output layer weights
    delta_output = error * sigmoid_derivative(output)
    # Calculate how much to adjust hidden layer weights
    delta_layer1 = delta_output.dot(weights2.T) * sigmoid_derivative(layer1)
    
    # Update weights
    weights2 += layer1.T.dot(delta_output) * learning_rate
    weights1 += X.T.dot(delta_layer1) * learning_rate
    
    # Print progress
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Error = {mean_error:.4f}")

# ----- PART 4: VISUALIZING RESULTS -----

# Create a grid to visualize the decision boundary
resolution = 50
glucose_range = np.linspace(0, 1, resolution)  # Normalized glucose values
bmi_range = np.linspace(0, 1, resolution)      # Normalized BMI values
grid = np.array([[g, b] for g in glucose_range for b in bmi_range])

# Get predictions for all points in the grid
_, predictions = forward(grid, weights1, weights2)
predictions = predictions.reshape(resolution, resolution)

# Create plot
plt.figure(figsize=(10, 6))

# Plot decision boundary
X_grid, Y_grid = np.meshgrid(glucose_range, bmi_range)
plt.contourf(X_grid, Y_grid, predictions.T, alpha=0.3, cmap=plt.cm.viridis)

# Plot the training points
diabetic = y == 1
plt.scatter(X[diabetic, 0], X[diabetic, 1], 
            color='red', edgecolor='k', s=100, marker='o', label='Diabetic')
plt.scatter(X[~diabetic, 0], X[~diabetic, 1], 
            color='blue', edgecolor='k', s=100, marker='o', label='Healthy')

# Add labels
plt.title('Neural Network Diabetes Prediction')
plt.xlabel('Glucose Level (normalized)')
plt.ylabel('BMI (normalized)')
plt.legend()
plt.grid(True)
plt.colorbar(label='Probability of Diabetes')

# ----- PART 5: TESTING THE NEURAL NETWORK -----

# Test with a few new patients
test_data = np.array([
    [0.6, 0.7],  # High glucose, high BMI - likely diabetic
    [0.1, 0.2],  # Low glucose, low BMI - likely healthy
    [0.4, 0.5],  # Medium values - less certain
])

# Convert back to original scale for display
test_glucose_orig = test_data[:, 0] * (200 - 70) + 70
test_bmi_orig = test_data[:, 1] * (50 - 20) + 20

# Make predictions
_, predictions = forward(test_data, weights1, weights2)

# Mark test points on the plot
plt.scatter(test_data[:, 0], test_data[:, 1], 
            color='yellow', edgecolor='black', s=200, marker='*', label='Test Patients')
plt.legend()
plt.savefig('diabetes_prediction.png')
plt.show()

# Display results
print("\nTesting the trained neural network on new patients:")
print("-" * 60)
print("  Glucose   BMI     Probability   Predicted")
print("-" * 60)
for i in range(len(test_data)):
    glucose = test_glucose_orig[i]
    bmi = test_bmi_orig[i]
    prob = predictions[i][0]
    predicted = "Diabetic" if prob > 0.5 else "Healthy"
    print(f"{glucose:7.1f}  {bmi:6.1f}   {prob:10.4f}    {predicted}")

print("\nNeural Network Summary:")
print(f"- Input layer: {input_size} neurons (glucose and BMI)")
print(f"- Hidden layer: {hidden_size} neurons with sigmoid activation")
print(f"- Output layer: {output_size} neuron with sigmoid activation")
print(f"- Final error: {error_history[-1]:.4f}")
print("\nExplanation: This neural network learns to classify patients as diabetic or")
print("healthy based on glucose levels and BMI. The visualization shows the")
print("decision boundary learned by the network.") 