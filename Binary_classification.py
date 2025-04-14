import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

# === Step 1: Load MNIST data (filter only 0s and 1s) ===
transform = transforms.ToTensor()

# Download training and test data
train_data_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data_full = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Filter only digits 0 and 1
train_data = [d for d in train_data_full if d[1] in [0, 1]]
test_data = [d for d in test_data_full if d[1] in [0, 1]]

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# === Step 2: Define Neural Network ===
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),            # 28x28 → 784
            nn.Linear(784, 128),     # Hidden layer
            nn.ReLU(),
            nn.Linear(128, 1),       # Output layer (1 neuron)
            nn.Sigmoid()             # Probability output
        )

    def forward(self, x):
        return self.model(x)

model = ImageClassifier()

# === Step 3: Loss and Optimizer ===
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# === Step 4: Training Loop ===
for epoch in range(5):
    total_loss = 0
    for X_batch, y_batch in train_loader:
        y_batch = y_batch.float().unsqueeze(1)  # Shape: [batch, 1]
        preds = model(X_batch)

        loss = loss_fn(preds, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# === Step 5: Testing ===
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_batch = y_batch.float().unsqueeze(1)
        preds = model(X_batch)
        predicted = (preds >= 0.5).float()
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

accuracy = correct / total * 100
print(f"\nTest Accuracy: {accuracy:.2f}%")

# === Step 6: Visualize a few test predictions ===
samples = next(iter(test_loader))
images, labels = samples
outputs = model(images).detach()
predicted = (outputs >= 0.5).float()

plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(images[i][0], cmap="gray")
    plt.title(f"Pred: {int(predicted[i].item())}")
    plt.axis("off")
plt.show()

from PIL import Image, ImageOps
import torchvision.transforms as transforms
import torch

def predict_custom_image(image_path, model):
    # === Load and preprocess image ===
    img = Image.open(image_path).convert('L')           # Convert to grayscale
    img = ImageOps.invert(img)                          # Invert: white digit on black bg
    img = img.resize((28, 28))                          # Resize to 28x28
    transform = transforms.ToTensor()                   # Convert to tensor
    img_tensor = transform(img).unsqueeze(0)            # Add batch dimension [1, 1, 28, 28]

    # === Predict ===
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        prediction = (output >= 0.5).float().item()

    print(f"Prediction: {output.item():.4f} → Classified as: {int(prediction)}")

# === Example usage ===
predict_custom_image("my_digit.jpg", model)
