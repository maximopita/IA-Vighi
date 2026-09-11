# entrenar_benignidad.py
# Entrena un clasificador de BENIGNIDAD (Benigno / Maligno) para UN organo,
# usando PHIKON (owkin/phikon) como extractor de caracteristicas de histopatologia
# y una capa lineal encima que aprende a distinguir las clases.
#
# Lee las carpetas dentro de "Datos/<Organo>/" (cada subcarpeta = una clase)
# y guarda el modelo en "modelo_<organo>.pth".
#
# Uso:
#   python entrenar_benignidad.py Pulmon
#   (si no pasas nada, usa "Pulmon" por defecto)

import os
import sys
import time
import torch
from torch import nn, optim
from PIL import Image

import motor

# ---- Que organo entrenar ----
ORGANO = sys.argv[1] if len(sys.argv) > 1 else "Pulmon"
CARPETA_DATOS = os.path.join("Datos", ORGANO)
ARCHIVO_MODELO = f"modelo_{ORGANO.lower()}.pth"

EPOCAS = 60          # el clasificador es liviano: muchas epocas son rapidas
TAM_LOTE = 16        # imagenes por lote al extraer features con Phikon
PROP_VALIDACION = 0.2
EXTENSIONES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

# ---- Revisar que haya fotos ----
if not os.path.isdir(CARPETA_DATOS):
    print(f"[!] No existe la carpeta '{CARPETA_DATOS}'.")
    print(f"    Crea 'Datos/{ORGANO}/Benigno' y 'Datos/{ORGANO}/Maligno' con imagenes.")
    raise SystemExit

# Cada subcarpeta es una clase.
clases = sorted([d for d in os.listdir(CARPETA_DATOS)
                 if os.path.isdir(os.path.join(CARPETA_DATOS, d))])
if not clases:
    print(f"[!] No hay subcarpetas de clases en 'Datos/{ORGANO}/'.")
    raise SystemExit

# Juntamos las rutas de todas las imagenes con su etiqueta.
rutas, etiquetas = [], []
for idx, clase in enumerate(clases):
    carpeta = os.path.join(CARPETA_DATOS, clase)
    fotos = [f for f in os.listdir(carpeta) if f.lower().endswith(EXTENSIONES)]
    print(f"  {clase}: {len(fotos)} fotos")
    for f in fotos:
        rutas.append(os.path.join(carpeta, f))
        etiquetas.append(idx)

if not rutas:
    print(f"\n[!] No hay fotos en 'Datos/{ORGANO}/'. Carga imagenes y volve a correr.")
    raise SystemExit

print(f"\nOrgano: {ORGANO}")
print("Clases detectadas:", clases)
print("Total de fotos:", len(rutas))

# ---- Extraer features con Phikon (una sola vez por imagen) ----
print("\nCargando Phikon y extrayendo caracteristicas (esto puede tardar)...")
t0 = time.time()
features = []
for i in range(0, len(rutas), TAM_LOTE):
    lote_rutas = rutas[i:i + TAM_LOTE]
    imagenes = [Image.open(r).convert("RGB") for r in lote_rutas]
    feats = motor.extraer_features(imagenes)   # [lote, 768]
    features.append(feats)
    print(f"  {min(i + TAM_LOTE, len(rutas))}/{len(rutas)} imagenes", end="\r")
X = torch.cat(features, dim=0)                 # [N, 768]
y = torch.tensor(etiquetas)
print(f"\nFeatures listas en {time.time() - t0:.0f}s. Shape: {tuple(X.shape)}")

# ---- Separar train / validacion ----
n_val = int(len(X) * PROP_VALIDACION)
generador = torch.Generator().manual_seed(42)
perm = torch.randperm(len(X), generator=generador)
idx_val, idx_train = perm[:n_val], perm[n_val:]
X_train, y_train = X[idx_train], y[idx_train]
X_val, y_val = X[idx_val], y[idx_val]
print(f"Entrenamiento: {len(X_train)} fotos | Validacion: {len(X_val)} fotos")

# ---- Entrenar el clasificador liviano sobre las features ----
clasificador = motor.crear_clasificador(clases)
criterio = nn.CrossEntropyLoss()
optimizador = optim.Adam(clasificador.parameters(), lr=0.001)

print("\nEntrenando el clasificador...\n")
for epoca in range(EPOCAS):
    clasificador.train()
    optimizador.zero_grad()
    salida = clasificador(X_train)
    perdida = criterio(salida, y_train)
    perdida.backward()
    optimizador.step()

    if (epoca + 1) % 10 == 0 or epoca == 0:
        clasificador.eval()
        with torch.no_grad():
            prec_train = (clasificador(X_train).argmax(1) == y_train).float().mean().item() * 100
            prec_val = (clasificador(X_val).argmax(1) == y_val).float().mean().item() * 100 if n_val else 0
        print(f"Epoca {epoca + 1}/{EPOCAS} - perdida: {perdida.item():.3f} "
              f"- precision train: {prec_train:.1f}% - precision validacion: {prec_val:.1f}%")

# ---- Guardar ----
torch.save({
    "estado": clasificador.state_dict(),
    "clases": clases,
    "organo": ORGANO,
    "base": motor.MODELO_BASE,     # que motor de features usa este modelo
}, ARCHIVO_MODELO)
print(f"\nListo. Modelo guardado en '{ARCHIVO_MODELO}'")
print(f"Motor base: {motor.MODELO_BASE} | Clases: {clases}")
