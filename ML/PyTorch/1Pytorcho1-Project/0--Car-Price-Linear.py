import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Дані
x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = torch.tensor([[400.0], [350.0], [300.0], [250.0], [200.0]])
# Модель лінійної регресії
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # Один вхідний і один вихідний параметр

    def forward(self, x):
        return self.linear(x)

# Ініціалізація моделі, функції втрат та оптимізатора
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Навчання моделі
num_epochs = 1000
for epoch in range(num_epochs):
    # Прямий прохід
    outputs = model(x)
    loss = criterion(outputs, y)

    # Зворотний прохід та оптимізація
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Виведення втрат кожні 100 епох
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Виведення результатів
print("Training complete.")
print(f'Final parameters: weight = {model.linear.weight.item():.4f}, bias = {model.linear.bias.item():.4f}')

# Передбачення ціни на 10-й рік
year_10 = torch.tensor([[10.0]])
predicted_price_10 = model(year_10).item()
print(f'Predicted price for 10th year: {predicted_price_10:.2f} тис. грн')

# Візуалізація результатів
# Додамо нові точки для побудови довшого графіка
x_extended = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]])
predicted_extended = model(x_extended).detach().numpy()
plt.plot(x.numpy(), y.numpy(), 'ro', label='Original data')
plt.plot(x_extended.numpy(), predicted_extended, label='Fitted line')
plt.xlabel('Вік автомобіля (роки)')
plt.ylabel('Ціна автомобіля (тис. грн)')
plt.legend()
plt.show()