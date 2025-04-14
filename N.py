import torch
import torch.nn as nn

# === Generate training data ===
# Numbers near 0 → label 0, near 1 → label 1
X_train = torch.tensor([[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]], dtype=torch.float32)
y_train = torch.tensor([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]], dtype=torch.float32)

# === Define the Neural Network ===
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, 4)  # 1 input → 4 neurons
        self.out = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.sigmoid(self.out(x))
        return x

model = BinaryClassifier()

# === Loss and Optimizer ===
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# === Training loop ===
for epoch in range(1000):
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# === User input and prediction ===
while True:
    user_input = input("\nEnter a number between 0 and 1 (or 'q' to quit): ")
    if user_input.lower() == 'q':
        break

    try:
        num = float(user_input)
        if 0 <= num <= 1:
            input_tensor = torch.tensor([[num]], dtype=torch.float32)
            prediction = model(input_tensor).item()
            result = 1 if prediction >= 0.5 else 0
            print(f"Prediction: {prediction:.4f} → Classified as: {result}")
        else:
            print("❌ Please enter a number between 0 and 1.")
    except ValueError:
        print("❌ Invalid input. Please enter a number.")
