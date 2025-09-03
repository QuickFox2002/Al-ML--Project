import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

print("🔄 Training model...")

X = torch.randn(200, 3, 128, 128)
y = torch.randint(0, 3, (200,))

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = SimpleCNN(num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_history = []
for epoch in range(5):
    running_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(dataloader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "plant_model.pt")
print("✅ Model saved as plant_model.pt")

plt.plot(loss_history, marker="o", color="blue")
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

print("\n📊 Evaluating model...")

model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for inputs, labels in dataloader:
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())

acc = accuracy_score(y_true, y_pred)
print(f"Model Accuracy: {acc:.2f}")

print("\n📑 Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Healthy", "Diseased", "Unknown"]))

print("\n🔮 Running prediction...")

model = SimpleCNN(num_classes=3)
model.load_state_dict(torch.load("plant_model.pt"))
model.eval()

test_image = torch.randn(1, 3, 128, 128)

with torch.no_grad():
    outputs = model(test_image)
    probs = F.softmax(outputs, dim=1).numpy().flatten()
    predicted = np.argmax(probs)

classes = ["Healthy", "Diseased", "Unknown"]
print(f"Predicted Class: {classes[predicted]}")

plt.bar(classes, probs, color=["green", "red", "Blue"])
plt.title("Prediction Probabilities")
plt.ylabel("Probability")
plt.show()
