import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from mini_alexnet import MiniAlexNet
import torch.nn.functional as F

# Clases
classes = ['paper', 'rock', 'scissors']

# Modelo
model = MiniAlexNet()
model.load_state_dict(torch.load("mini_alexnet_rps.pth", map_location=torch.device('cpu')))
model.eval()

# Transformación
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Interfaz
st.title("Rock-Paper-Scissors Classifier")
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    st.markdown(f"### Predicción: `{classes[predicted.item()]}`")
    st.markdown(f"### Confianza: `{confidence.item() * 100:.2f}%`")