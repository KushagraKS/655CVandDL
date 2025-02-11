import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

x = np.linspace(-6, 6, 400)
y_sigmoid = sigmoid(x)
y_derivative = sigmoid_derivative(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y_sigmoid, label='Sigmoid Function', color='blue')
plt.plot(x, y_derivative, label='Sigmoid Derivative', color='red', linestyle='dashed')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Sigmoid Function and Its Derivative')
plt.legend()
plt.grid()
plt.show()
