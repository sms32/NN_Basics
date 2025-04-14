import numpy as np
import matplotlib.pyplot as plt

# Activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([[5], [7], [9], [11], [13]])

# Parameters
np.random.seed(0)
input_dim = 1
hidden1_dim = 2
hidden2_dim = 2
output_dim = 1
lr = 0.01
epochs = 1000

# Weights and biases
W1 = np.random.randn(input_dim, hidden1_dim)
b1 = np.zeros((1, hidden1_dim))

W2 = np.random.randn(hidden1_dim, hidden2_dim)
b2 = np.zeros((1, hidden2_dim))

W3 = np.random.randn(hidden2_dim, output_dim)
b3 = np.zeros((1, output_dim))

# Loss tracker
losses = []

# Training loop
for epoch in range(epochs):
    # === Forward Pass ===
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    Z3 = np.dot(A2, W3) + b3
    y_pred = Z3

    # === Loss ===
    loss = np.mean((y - y_pred) ** 2)
    losses.append(loss)

    # === Backward Pass ===
    dZ3 = 2 * (y_pred - y) / y.shape[0]
    dW3 = np.dot(A2.T, dZ3)
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * sigmoid_derivative(Z2)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(Z1)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # === Update Weights ===
    W3 -= lr * dW3
    b3 -= lr * db3
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Final prediction
print("\nPredictions after training:")
print(y_pred)

# Plot loss
plt.plot(losses)
plt.title("Loss Curve (Two Hidden Layers)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
