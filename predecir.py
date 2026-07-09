# predecir.py
# Usa el modelo entrenado (modelo_organos.pth) para adivinar que organo
# aparece en una imagen.
#
# Uso:
#   python predecir.py ruta/a/la/foto.jpg

import sys
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

ARCHIVO_MODELO = "modelo_organos.pth"

# ---- Leer la ruta de la imagen ----
if len(sys.argv) < 2:
    print("Uso: python predecir.py ruta/a/la/foto.jpg")
    raise SystemExit

ruta_imagen = sys.argv[1]

# ---- Cargar el modelo entrenado ----
datos_guardados = torch.load(ARCHIVO_MODELO, weights_only=False)
clases = datos_guardados["clases"]

modelo = models.resnet18()
modelo.fc = nn.Linear(modelo.fc.in_features, len(clases))
modelo.load_state_dict(datos_guardados["estado"])
modelo.eval()

# ---- Preparar la imagen (igual que en el entrenamiento, pero sin giros) ----
transformacion = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

imagen = Image.open(ruta_imagen).convert("RGB")
entrada = transformacion(imagen).unsqueeze(0)

# ---- Predecir ----
with torch.no_grad():
    salida = modelo(entrada)
    probabilidades = torch.nn.functional.softmax(salida[0], dim=0)

indice = probabilidades.argmax().item()
confianza = probabilidades[indice].item() * 100

print("\n---------------------")
print("Imagen:", ruta_imagen)
print("Organo detectado:", clases[indice])
print("Confianza:", round(confianza, 2), "%")

# Mostrar tambien el ranking completo
print("\nTodas las probabilidades:")
for i, clase in enumerate(clases):
    print(f"  {clase}: {probabilidades[i].item() * 100:.1f}%")
