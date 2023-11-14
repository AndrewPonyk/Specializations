import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("Hello, linear regression in tensorflow")
# Generate some sample data for x and y
x = np.array([1, 2, 3, 4, 5, 6]).reshape((-1, 1))
y = np.array([3, 4, 6, 7, 8, 19])


# Define the model inputs and outputs
inputs = tf.keras.layers.Input(shape=(1,))
outputs = tf.keras.layers.Dense(units=1)(inputs)

# Define the model and compile it
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='sgd', loss='mse')

# Fit the model to the data
model.fit(x, y, epochs=100)

# Predict the value of y for a new value of x
x_new = np.array([[7]])
y_new = model.predict(x_new)
print(y_new)

# Plot the data and the regression line
plt.scatter(x, y)
plt.plot(x, model.predict(x), color='red')
plt.show()
s = "1234"
print(s)