import numpy as np

# Activation function (Sigmoid) and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - x ** 2


# Training data: OR
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [0],
              [0],
              [1]])

# Seed for reproducibility
np.random.seed(42)

# Network architecture
input_size = 2
hidden_size = 4
output_size = 1

# Initialize weights and biases
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Training parameters
epochs = 10000
lr = 0.1

for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = tanh(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = tanh(z2)

    # Compute loss (mean squared error)
    loss = np.mean((y - a2) ** 2)

    # Backpropagation
    d_a2 = (a2 - y) * tanh_derivative(a2)
    d_W2 = np.dot(a1.T, d_a2)
    d_b2 = np.sum(d_a2, axis=0, keepdims=True)

    d_a1 = np.dot(d_a2, W2.T) * tanh_derivative(a1)
    d_W1 = np.dot(X.T, d_a1)
    d_b1 = np.sum(d_a1, axis=0, keepdims=True)

    # Update weights and biases
    W2 -= lr * d_W2
    b2 -= lr * d_b2
    W1 -= lr * d_W1
    b1 -= lr * d_b1

    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} Loss: {loss:.4f}")

# Final prediction
a2 = np.round(a2)  # Round to get values like 0 or 1
a2 = a2.astype(int)  # Convert the entire array to integers

print("\nFinal Output:")
print(a2)  # Output the final prediction array

