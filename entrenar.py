# entrenar.py
# Entrena un clasificador de organos usando transfer learning con ResNet18.
# Lee las carpetas dentro de "Organos/" (cada carpeta = un organo) y guarda
# el modelo entrenado en "modelo_organos.pth".

import os
import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

# ---- Configuracion ----
CARPETA_DATOS = "Organos"
ARCHIVO_MODELO = "modelo_organos.pth"
EPOCAS = 10          # cuantas veces recorre todas las fotos (subilo si tenes muchas fotos)
TAM_LOTE = 8         # cuantas fotos procesa a la vez

# ---- Preparar las imagenes ----
# Usamos las mismas transformaciones que espera ResNet18, mas un poco de
# "data augmentation" (giros/recortes) para que aprenda mejor con pocas fotos.
transformacion = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Revisamos que haya fotos antes de arrancar, asi damos un mensaje claro
# en vez del error tecnico de ImageFolder.
EXTENSIONES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
total_fotos = 0
if os.path.isdir(CARPETA_DATOS):
    for carpeta in os.listdir(CARPETA_DATOS):
        ruta_carpeta = os.path.join(CARPETA_DATOS, carpeta)
        if os.path.isdir(ruta_carpeta):
            fotos = [f for f in os.listdir(ruta_carpeta) if f.lower().endswith(EXTENSIONES)]
            print(f"  {carpeta}: {len(fotos)} fotos")
            total_fotos += len(fotos)

if total_fotos == 0:
    print("\n[!] No hay ninguna foto todavia.")
    print("    Pone imagenes dentro de las carpetas de 'Organos/' y volve a correr.")
    print("    Ejemplo: Organos/Corazon/corazon1.jpg")
    raise SystemExit

datos = datasets.ImageFolder(CARPETA_DATOS, transform=transformacion)
cargador = torch.utils.data.DataLoader(datos, batch_size=TAM_LOTE, shuffle=True)

clases = datos.classes
print("Organos detectados:", clases)
print("Total de fotos:", len(datos))

if len(datos) == 0:
    print("\n[!] No hay fotos. Pone imagenes dentro de las carpetas de Organos/ y volve a correr.")
    raise SystemExit

# ---- Cargar ResNet18 pre-entrenada y adaptarla ----
weights = ResNet18_Weights.DEFAULT
modelo = models.resnet18(weights=weights)

# Congelamos todas las capas: no queremos re-entrenar lo que ya sabe.
for parametro in modelo.parameters():
    parametro.requires_grad = False

# Reemplazamos la ultima capa por una nueva del tamano de nuestras clases.
# Esta capa SI se entrena (es la que aprende a distinguir organos).
modelo.fc = nn.Linear(modelo.fc.in_features, len(clases))

# ---- Entrenamiento ----
criterio = nn.CrossEntropyLoss()
optimizador = optim.Adam(modelo.fc.parameters(), lr=0.001)

modelo.train()
for epoca in range(EPOCAS):
    perdida_total = 0.0
    aciertos = 0
    for imagenes, etiquetas in cargador:
        optimizador.zero_grad()
        salida = modelo(imagenes)
        perdida = criterio(salida, etiquetas)
        perdida.backward()
        optimizador.step()

        perdida_total += perdida.item()
        aciertos += (salida.argmax(1) == etiquetas).sum().item()

    precision = 100 * aciertos / len(datos)
    print(f"Epoca {epoca + 1}/{EPOCAS} - perdida: {perdida_total:.3f} - precision: {precision:.1f}%")

# ---- Guardar el modelo y los nombres de las clases ----
torch.save({"estado": modelo.state_dict(), "clases": clases}, ARCHIVO_MODELO)
print(f"\nListo. Modelo guardado en '{ARCHIVO_MODELO}'")
