# nucleos.py  --  Motor de analisis a nivel CELULAR (nucleo por nucleo).
#
# Detecta y clasifica cada nucleo con HoVer-Net (pesos PanNuke, via TIAToolbox) y
# dibuja su contorno coloreado por tipo. La app (app.py) lo llama como un proceso
# aparte, porque corre en su propio entorno (Python 3.12, ver README).
#
# Uso suelto (compara original y marcada lado a lado):
#   C:\vighi-venvs\celular\Scripts\python.exe motor_celular\nucleos.py <imagen> [...]
# Uso desde la app (un solo archivo de salida + progreso):
#   ... nucleos.py <imagen> --png marcada.png --json resumen.json --progreso prog.json
#
# Licencia de los pesos PanNuke: CC BY-NC-SA (solo uso no comercial).

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

from tiatoolbox.models.architecture import get_pretrained_model

MODELO = "hovernet_fast-pannuke"
ENTRADA, SALIDA = 256, 164            # HoVer-Net fast: entra 256x256, sale 164x164 central
MARGEN = (ENTRADA - SALIDA) // 2      # 46 px
MAX_LADO = 1536                       # imagenes mas grandes se reducen (acota el tiempo)
TIPOS = {1: "Neoplasico", 2: "Inflamatorio", 3: "Conectivo", 4: "Muerto",
         5: "Epitelial no neoplasico"}
COLORES = {1: (230, 25, 30), 2: (40, 170, 70), 3: (40, 100, 230),
           4: (240, 190, 20), 5: (0, 190, 210)}


def cargar_modelo():
    modelo, _ = get_pretrained_model(MODELO)
    modelo.eval()
    return modelo


def _es_fondo(region_rgb, umbral=0.82):
    """True si la region es mayormente fondo (vidrio/blanco): no vale la pena analizarla."""
    g = region_rgb.astype(np.float32).mean(axis=2) / 255.0
    return g.mean() > umbral and g.std() < 0.06


@torch.no_grad()
def mapas_crudos(modelo, img_rgb, tam_lote=4, progreso=None):
    """Corre el modelo por ventanas y arma los 3 mapas (np, hv, tp) de la imagen
    completa, para hacer UN solo post-proceso y no duplicar nucleos en los bordes.
    Las ventanas que son solo fondo se saltean (ahorra mucho tiempo en imagenes con
    grasa o vidrio). Devuelve (mapas o None si no hay tejido, total, analizadas)."""
    H, W = img_rgb.shape[:2]
    ny, nx = math.ceil(H / SALIDA), math.ceil(W / SALIDA)
    pad_b = ny * SALIDA - H
    pad_r = nx * SALIDA - W
    p = np.pad(img_rgb, ((MARGEN, MARGEN + pad_b), (MARGEN, MARGEN + pad_r), (0, 0)),
               mode="reflect")
    todas = [(iy * SALIDA, ix * SALIDA) for iy in range(ny) for ix in range(nx)]
    coords = [(y, x) for y, x in todas if not _es_fondo(img_rgb[y:y + SALIDA, x:x + SALIDA])]
    if not coords:
        return None, len(todas), 0

    completos = None
    for k in range(0, len(coords), tam_lote):
        lote = coords[k:k + tam_lote]
        batch = np.stack([p[y:y + ENTRADA, x:x + ENTRADA] for y, x in lote])
        salida = modelo.infer_batch(modelo, torch.from_numpy(batch), device="cpu")
        if completos is None:
            completos = [np.zeros((ny * SALIDA, nx * SALIDA, s.shape[-1]), dtype=np.float32)
                         for s in salida]
        for j, (y, x) in enumerate(lote):
            for m, s in zip(completos, salida):
                m[y:y + SALIDA, x:x + SALIDA] = s[j]
        if progreso:
            progreso(min(k + tam_lote, len(coords)), len(coords))
    return [m[:H, :W] for m in completos], len(todas), len(coords)


def analizar_imagen(modelo, img, max_lado=MAX_LADO, progreso=None):
    """Analiza una imagen PIL. Devuelve (imagen_marcada PIL, resumen dict)."""
    img = img.convert("RGB")
    reducida_de = None
    if max(img.size) > max_lado:
        reducida_de = list(img.size)
        s = max_lado / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    arr = np.array(img)

    t = time.time()
    raw, n_ventanas, n_analizadas = mapas_crudos(modelo, arr, progreso=progreso)
    conteo = {n: 0 for n in TIPOS.values()}
    marcada = img.copy()
    if raw is not None:
        res = modelo.postproc(raw)
        # TIAToolbox 2.x: dict con 'info_dict' en COLUMNAS (box, centroid, contours, prob, type)
        info = (res[0] if isinstance(res, tuple) else res)["info_dict"]
        tipos, contornos, cajas = info["type"], info["contours"], info["box"]
        dib = ImageDraw.Draw(marcada)
        for i in range(len(tipos)):
            tipo = int(np.asarray(tipos[i]).reshape(-1)[0])
            if tipo not in TIPOS:
                continue
            conteo[TIPOS[tipo]] += 1
            # 'contours' es un arreglo (N, puntos_max, 2): cada contorno esta RELLENADO
            # hasta el largo maximo. Descartamos el relleno quedandonos con los puntos
            # que caen dentro de la caja del nucleo (x0, y0, x1, y1).
            c = np.asarray(contornos[i])
            x0, y0, x1, y1 = np.asarray(cajas[i]).tolist()
            dentro = (c[:, 0] >= x0 - 1) & (c[:, 0] <= x1 + 1) & \
                     (c[:, 1] >= y0 - 1) & (c[:, 1] <= y1 + 1)
            cont = [(int(q[0]), int(q[1])) for q in c[dentro]]
            if len(cont) >= 3:
                dib.line(cont + [cont[0]], fill=COLORES[tipo], width=2)

    total = sum(conteo.values())
    resumen = {
        "tam": list(img.size), "reducida_de": reducida_de,
        "total": total, "conteo": conteo,
        "frac_neoplasico": (conteo["Neoplasico"] / total) if total else 0.0,
        "segundos": round(time.time() - t, 1),
        "ventanas": n_ventanas, "ventanas_analizadas": n_analizadas,
    }
    return marcada, resumen


def _escribir_json(ruta, obj):
    """Escritura atomica (la app lee el archivo mientras el proceso corre)."""
    tmp = ruta + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, ruta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("imagenes", nargs="+")
    ap.add_argument("--max-lado", type=int, default=MAX_LADO)
    ap.add_argument("--png", help="(modo app) ruta de la imagen marcada de salida")
    ap.add_argument("--json", help="(modo app) ruta del resumen JSON de salida")
    ap.add_argument("--progreso", help="(modo app) archivo JSON donde se va informando el avance")
    ap.add_argument("--salida", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "salidas"),
                    help="(modo suelto) carpeta de salida")
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    def avance(hecho, total):
        if a.progreso:
            _escribir_json(a.progreso, {"hecho": hecho, "total": total})

    modelo = cargar_modelo()

    if a.png:                                   # modo app: una imagen, salidas por archivo
        marcada, resumen = analizar_imagen(modelo, Image.open(a.imagenes[0]), a.max_lado, avance)
        marcada.save(a.png)
        if a.json:
            _escribir_json(a.json, resumen)
    else:                                       # modo suelto: original y marcada lado a lado
        os.makedirs(a.salida, exist_ok=True)
        for r in a.imagenes:
            orig = Image.open(r).convert("RGB")
            marcada, res = analizar_imagen(modelo, orig, a.max_lado)
            base = marcada.size
            lado = Image.new("RGB", (base[0] * 2 + 10, base[1]), "white")
            lado.paste(orig.resize(base), (0, 0)); lado.paste(marcada, (base[0] + 10, 0))
            partes = os.path.normpath(r).split(os.sep)
            nombre = "_".join(partes[-3:-1] + [os.path.splitext(partes[-1])[0]]) if len(partes) >= 3 \
                else os.path.splitext(partes[-1])[0]
            salida = os.path.join(a.salida, nombre + "_nucleos.png")
            lado.save(salida)
            print(f"{os.path.basename(r)} {res['tam']}: {res['total']} nucleos en {res['segundos']}s "
                  f"({res['ventanas_analizadas']}/{res['ventanas']} ventanas) | "
                  f"neoplasicos {res['conteo']['Neoplasico']} ({res['frac_neoplasico'] * 100:.0f}%) "
                  f"-> {salida}", flush=True)
