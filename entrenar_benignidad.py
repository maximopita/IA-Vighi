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

# ---- Argumentos: organo y banderas ----
#   python entrenar_benignidad.py Mama                 -> recortes puros (clasico)
#   python entrenar_benignidad.py Mama --panoramica    -> corta panoramicas en parches
#   python entrenar_benignidad.py Mama --panoramica --parche=256
argumentos = [a for a in sys.argv[1:] if not a.startswith("--")]
banderas = [a for a in sys.argv[1:] if a.startswith("--")]
ORGANO = argumentos[0] if argumentos else "Pulmon"
PANORAMICA = "--panoramica" in banderas
PARCHE_PX = 224
for b in banderas:
    if b.startswith("--parche="):
        PARCHE_PX = int(b.split("=")[1])

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

# ---- Extraer features con Phikon ----
print("\nCargando Phikon y extrayendo caracteristicas (esto puede tardar)...")
t0 = time.time()

if PANORAMICA:
    # Cada imagen (panoramica) se corta en parches; cada parche hereda la etiqueta
    # de su carpeta. Asi el modelo aprende a la MISMA escala en que despues analiza.
    print(f"Modo PANORAMICA: corto cada imagen en parches de {PARCHE_PX}px "
          f"(descarto fondo).")
    feats_list, etq_list = [], []
    for n, (r, lab) in enumerate(zip(rutas, etiquetas), 1):
        img = Image.open(r).convert("RGB")
        parches = motor.recortar_en_parches(img, tam=PARCHE_PX, solapamiento=0.25)
        for k in range(0, len(parches), TAM_LOTE):
            fe = motor.extraer_features(parches[k:k + TAM_LOTE])
            feats_list.append(fe)
            etq_list.extend([lab] * fe.shape[0])
        print(f"  {n}/{len(rutas)} imagenes -> {len(etq_list)} parches", end="\r")
    X = torch.cat(feats_list, dim=0)
    y = torch.tensor(etq_list)
else:
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

# ---- Prototipo de organo (para verificar que la imagen sea de este organo) ----
# Es el "centro" del organo. Se calcula SIEMPRE con las imagenes ENTERAS (la identidad
# del organo es una propiedad global), aunque el clasificador se haya entrenado con
# parches. Asi la verificacion de organo funciona igual en modo clasico y panoramico.
import torch.nn.functional as F
import random as _random
if PANORAMICA:
    feats_enteras = []
    for i in range(0, len(rutas), TAM_LOTE):
        ims = [Image.open(r).convert("RGB") for r in rutas[i:i + TAM_LOTE]]
        feats_enteras.append(motor.extraer_features(ims))
    Xw = torch.cat(feats_enteras, dim=0)
else:
    Xw = X   # en modo clasico X ya son features de imagenes enteras
prototipo = F.normalize(Xw.mean(0, keepdim=True), dim=1)[0]   # vector [768]

# Segundo prototipo a OTRA escala, para reconocer el organo en recortes Y en campos
# amplios (si no, un colon ancho se confunde con un organo entrenado en panoramicas).
# Se decide por el TAMANO REAL de las imagenes (no por la bandera 'panoramica').
_random.seed(0)
_sin_fb = [r for r in rutas if "feedback" not in r.lower()]
_otros_feats = []
_muestra = Image.open(_sin_fb[0]); _es_ancho = max(_muestra.size) >= 1200
if _es_ancho:
    # imagenes anchas (panoramicas reales) -> patron nativo 'ancho'; agregamos 'chico'.
    for p in _sin_fb[:40]:
        im = Image.open(p).convert("RGB"); W, H = im.size
        for _ in range(4):
            if W > 768 and H > 768:
                x = _random.randint(0, W - 768); y = _random.randint(0, H - 768)
                _otros_feats.append(im.crop((x, y, x + 768, y + 768)))
            else:
                _otros_feats.append(im.resize((768, 768)))
else:
    # ya tenemos patron 'chico' (tiles); agregamos uno 'ancho' con mosaicos 3x3.
    for _ in range(25):
        mos = Image.new("RGB", (1536, 1536))
        for k in range(9):
            c = Image.open(_random.choice(_sin_fb)).convert("RGB").resize((512, 512))
            mos.paste(c, ((k % 3) * 512, (k // 3) * 512))
        _otros_feats.append(mos)
_Xo = []
for i in range(0, len(_otros_feats), TAM_LOTE):
    _Xo.append(motor.extraer_features(_otros_feats[i:i + TAM_LOTE]))
prototipo_otro = F.normalize(torch.cat(_Xo, 0).mean(0, keepdim=True), dim=1)[0]
prototipos = [prototipo, prototipo_otro]   # [nativo, otra-escala]

# ---- Guardar ----
torch.save({
    "estado": clasificador.state_dict(),
    "clases": clases,
    "organo": ORGANO,
    "base": motor.MODELO_BASE,     # que motor de features usa este modelo
    "prototipo": prototipo,        # compat: centro del organo (escala nativa)
    "prototipos": prototipos,      # varios patrones (chico + ancho) para la verificacion
    "panoramica": PANORAMICA,      # si se entreno cortando panoramicas en parches
    "parche_px": PARCHE_PX if PANORAMICA else None,  # escala del parche
}, ARCHIVO_MODELO)
print(f"\nListo. Modelo guardado en '{ARCHIVO_MODELO}'")
print(f"Motor base: {motor.MODELO_BASE} | Clases: {clases}"
      + (f" | PANORAMICA parche={PARCHE_PX}px" if PANORAMICA else ""))
