import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from mini_alexnet import MiniAlexNet
import torch.nn as nn
import torch.optim as optim
import os

# Transformación
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Dataset
base_path = os.path.join(os.getcwd(), "Rock-Paper-Scissors")
train_set = ImageFolder(os.path.join(base_path, "train"), transform=transform)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(train_set, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniAlexNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento
for epoch in range(10):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# Validación final
model.eval()
correct = total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Precisión en validación: {correct / total:.2f}")

# Guardar modelo
torch.save(model.state_dict(), "mini_alexnet_rps.pth")
print("Modelo guardado como mini_alexnet_rps.pth")