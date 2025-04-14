import torch
import torch.nn as nn

# === Dummy Data: Learn to classify 0 or 1 ===
X = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)  # Inputs
y = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)  # Labels (0 or 1)

# === Step 1: Define the Neural Network ===
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(1, 2)  # input → hidden (1 input → 2 neurons)
        self.out = nn.Linear(2, 1)     # hidden → output (2 → 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = torch.relu(x)
        x = self.out(x)
        x = self.sigmoid(x)  # Output is probability
        return x

model = SimpleNN()

# === Step 2: Loss function and optimizer ===
loss_fn = nn.BCELoss()  # Binary Cross-Entropy
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# === Step 3: Training loop ===
epochs = 1000
for epoch in range(epochs):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()        # Backpropagation
    optimizer.step()       # Update weights

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# === Final prediction ===
print("\nPredictions:")
print(model(X).detach())
