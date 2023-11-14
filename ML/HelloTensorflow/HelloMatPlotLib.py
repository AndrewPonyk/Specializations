import matplotlib.pyplot as plt

# Define the x and y data
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

# Create a line plot
plt.plot(x, y)

# Add labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Line Plot')

# Show the plot
plt.show()