import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from mini_alexnet import MiniAlexNet
import numpy as np
import os

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_path = os.path.join(os.getcwd(), "Rock-Paper-Scissors")

# Transformación
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Dataset de test
test_set = ImageFolder(os.path.join(base_path, "test"), transform=transform)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
classes = test_set.classes  # ['paper', 'rock', 'scissors']

# Cargar el modelo entrenado
model = MiniAlexNet().to(device)
model.load_state_dict(torch.load("mini_alexnet_rps.pth", map_location=device))
model.eval()

# Evaluación
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Resultados
print("=== Reporte de Clasificación ===")
print(classification_report(all_labels, all_preds, target_names=classes))

print("\n=== Matriz de Confusión ===")
print(confusion_matrix(all_labels, all_preds))

accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"\nPrecisión Total en el set de test: {accuracy*100:.2f}%")