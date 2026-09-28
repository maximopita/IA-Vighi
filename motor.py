# motor.py
# Motor de IA compartido por el entrenamiento y la app.
#
# Usa PHIKON (owkin/phikon), un modelo pre-entrenado en histopatologia,
# como extractor de caracteristicas. Encima va un clasificador liviano
# (una capa lineal) que aprende Benigno / Maligno.
#
# Licencia de Phikon: "Owkin non-commercial license" (uso de investigacion
# e interno, NO comercial).

import io
import base64

import numpy as np
import torch
from torch import nn
from PIL import Image, ImageFilter
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

# Modelo base de patologia. ViT-B, devuelve vectores de 768 dimensiones.
MODELO_BASE = "owkin/phikon"
DIM_FEATURES = 768

# Segundo modelo, INDEPENDIENTE del primero: PLIP (CLIP entrenado en patologia).
# Lo usamos como "segunda opinion" benigno/maligno por similitud imagen-texto.
# No describe morfologia fina (en zero-shot no es confiable para eso); solo da
# una estimacion gruesa que sirve para ver si concuerda con Phikon.
MODELO_SEGUNDO = "vinid/plip"

# Verificacion de organo (que la imagen sea del organo elegido).
# Se compara la imagen contra el "prototipo" (centro) de cada organo en el espacio
# de features de Phikon, por similitud coseno.
UMBRAL_ORGANO = 0.45   # similitud minima con el organo elegido (piso; OOD por debajo)
UMBRAL_PROPIO = 0.55   # por encima de esto, es claramente el organo elegido
TEMP_ORGANO = 0.1      # temperatura para mostrar las similitudes como % legibles

# Traduccion de organos (ES -> EN) para armar los prompts de PLIP (entiende ingles).
_ORGANO_EN = {"pulmon": "lung", "colon": "colon", "mama": "breast",
              "prostata": "prostate", "piel": "skin", "rinon": "kidney",
              "higado": "liver", "estomago": "stomach", "tiroides": "thyroid"}

# Se cargan una sola vez (son pesados) y se reutilizan.
_procesador = None
_backbone = None
_plip_model = None
_plip_proc = None


def cargar_backbone():
    """Carga Phikon (procesador + red). La primera vez descarga ~335 MB.
    Usa attn_implementation='eager' para poder leer los mapas de atencion
    (necesarios para el 'porque' / mapa de calor)."""
    global _procesador, _backbone
    if _backbone is None:
        _procesador = AutoImageProcessor.from_pretrained(MODELO_BASE)
        _backbone = AutoModel.from_pretrained(MODELO_BASE, attn_implementation="eager")
        _backbone.eval()
    return _procesador, _backbone


def cargar_plip():
    """Carga PLIP (segundo modelo, independiente). La primera vez descarga ~600 MB."""
    global _plip_model, _plip_proc
    if _plip_model is None:
        _plip_model = CLIPModel.from_pretrained(MODELO_SEGUNDO)
        _plip_proc = CLIPProcessor.from_pretrained(MODELO_SEGUNDO)
        _plip_model.eval()
    return _plip_model, _plip_proc


def _prompts_benig_malig(organo):
    """Frases (en ingles, que es como PLIP entiende texto) que describen tejido
    benigno vs maligno para el organo.
    Estos prompts se eligieron midiendo el acierto sobre las imagenes reales del
    laboratorio: "{organo} adenocarcinoma" resulto el mejor balance (95% en colon
    y 95% en pulmon). Otros prompts genericos daban peor en algun organo."""
    en = _ORGANO_EN.get(organo.lower(), organo.lower())
    benignas = [f"benign {en} tissue"]
    malignas = [f"{en} adenocarcinoma"]
    return benignas, malignas


@torch.no_grad()
def segunda_opinion(imagen_pil, organo):
    """Segunda opinion INDEPENDIENTE (PLIP) benigno/maligno para 'organo'.
    Devuelve dict: {'prob_maligno': 0..1, 'prob_benigno': 0..1}.
    Es una estimacion gruesa por similitud imagen-texto; sirve para contrastar
    con el clasificador principal (concordancia)."""
    model, proc = cargar_plip()
    benignas, malignas = _prompts_benig_malig(organo)
    frases = benignas + malignas
    entradas = proc(text=frases, images=imagen_pil.convert("RGB"),
                    return_tensors="pt", padding=True)
    probs = model(**entradas).logits_per_image.softmax(dim=1)[0]
    n_ben = len(benignas)
    p_ben = float(probs[:n_ben].sum())
    p_mal = float(probs[n_ben:].sum())
    total = p_ben + p_mal + 1e-8
    return {"prob_benigno": p_ben / total, "prob_maligno": p_mal / total}


@torch.no_grad()
def extraer_features(imagenes_pil, tam_lote=16):
    """Recibe una imagen PIL o una lista de imagenes PIL.
    Devuelve un tensor [N, 768] con las caracteristicas de Phikon (token CLS).
    Procesa en tandas de 'tam_lote' para no reventar la memoria con panoramicas
    que generan muchos parches."""
    procesador, backbone = cargar_backbone()
    if not isinstance(imagenes_pil, (list, tuple)):
        imagenes_pil = [imagenes_pil]
    salidas = []
    for i in range(0, len(imagenes_pil), tam_lote):
        lote = imagenes_pil[i:i + tam_lote]
        entradas = procesador(lote, return_tensors="pt")
        cls = backbone(**entradas).last_hidden_state[:, 0, :]
        salidas.append(cls)
    return torch.cat(salidas, dim=0)


@torch.no_grad()
def features_y_atencion(imagen_pil):
    """Un solo forward que devuelve:
    - las features CLS [1, 768] (para clasificar)
    - un mapa de atencion [lado, lado] en [0,1] (attention rollout), que indica
      donde la red concentro su atencion."""
    procesador, backbone = cargar_backbone()
    entradas = procesador(imagen_pil, return_tensors="pt")
    salida = backbone(**entradas, output_attentions=True)
    cls = salida.last_hidden_state[:, 0, :]

    # Attention rollout: multiplicamos las atenciones de todas las capas.
    atenciones = salida.attentions              # tupla de [1, cabezas, N, N]
    n = atenciones[0].shape[-1]
    acumulado = torch.eye(n)
    for att in atenciones:
        a = att.mean(1)[0] + torch.eye(n)       # promedio de cabezas + residual
        a = a / a.sum(-1, keepdim=True)
        acumulado = a @ acumulado
    mask = acumulado[0, 1:]                      # atencion del token CLS a los parches
    lado = int(mask.shape[0] ** 0.5)
    mask = mask[:lado * lado].reshape(lado, lado)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return cls, mask


def _jet(v):
    """Colormap tipo 'jet': [0,1] -> RGB (azul->verde->amarillo->rojo)."""
    v = np.clip(v, 0, 1)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], -1)


def overlay_mapa(imagen_pil, mask, alpha=0.5, max_lado=512):
    """Superpone el mapa de atencion sobre la imagen. Devuelve una imagen PIL."""
    img = imagen_pil.convert("RGB")
    # Limitamos el tamano para que la respuesta no sea pesada.
    if max(img.size) > max_lado:
        escala = max_lado / max(img.size)
        img = img.resize((int(img.width * escala), int(img.height * escala)))

    W, H = img.size
    m = mask.numpy() ** 0.7                       # gamma: realza zonas medias
    m = Image.fromarray((m * 255).astype("uint8")).resize((W, H), Image.BICUBIC)
    m = np.array(m.filter(ImageFilter.GaussianBlur(6))) / 255.0

    heat = _jet(m)
    base = np.array(img).astype(float) / 255.0
    over = (1 - alpha * m[..., None]) * base + alpha * m[..., None] * heat
    return Image.fromarray((np.clip(over, 0, 1) * 255).astype("uint8"))


def _es_fondo(parche, umbral=0.82):
    """True si el parche es mayormente fondo (vidrio/blanco), no tejido.
    Sirve para no entrenar ni analizar zonas vacias de las panoramicas."""
    g = np.asarray(parche.convert("L"), dtype=float) / 255.0
    # Fondo = muy claro y sin apenas variacion (el tejido tiene textura y color).
    return g.mean() > umbral and g.std() < 0.06


def recortar_en_parches(imagen_pil, tam=224, solapamiento=0.0, saltar_fondo=True):
    """Corta una imagen (tipicamente una PANORAMICA) en parches cuadrados de 'tam'
    pixeles. Devuelve una lista de imagenes PIL. Descarta los parches de fondo.
    Si la imagen es mas chica que 'tam', devuelve la imagen entera."""
    img = imagen_pil.convert("RGB")
    W, H = img.size
    if W <= tam and H <= tam:
        return [img]
    paso = max(1, int(tam * (1 - solapamiento)))
    parches = []
    for y in range(0, max(1, H - tam + 1), paso):
        for x in range(0, max(1, W - tam + 1), paso):
            p = img.crop((x, y, min(x + tam, W), min(y + tam, H)))
            if saltar_fondo and _es_fondo(p):
                continue
            parches.append(p)
    return parches or [img]


@torch.no_grad()
def mapa_malignidad(imagen_pil, clasificador, idx_maligno, grid=6, tam=None):
    """Divide la imagen en una grilla y clasifica CADA region como benigna/maligna.
    Devuelve un tensor [filas, cols] con la probabilidad de MALIGNO de cada region.
    A diferencia del mapa de atencion (que muestra 'donde miro'), esto muestra
    'donde considera que hay cancer': una imagen benigna queda casi sin marcar.

    Si 'tam' (px por parche) se pasa, la grilla se calcula para que cada region
    mida ~tam pixeles (asi coincide con la escala con que se entreno la panoramica).
    Si no, usa una grilla fija de 'grid' x 'grid'."""
    img = imagen_pil.convert("RGB")
    W, H = img.size
    if tam:
        cols = max(1, round(W / tam)); filas = max(1, round(H / tam))
    else:
        cols = filas = grid
    cw, ch = W / cols, H / filas
    # Cada region se agranda medio casillero para darle contexto al clasificador.
    parches = []
    for i in range(filas):
        for j in range(cols):
            x0 = max(0, int(j * cw - cw / 2)); y0 = max(0, int(i * ch - ch / 2))
            x1 = min(W, int((j + 1) * cw + cw / 2)); y1 = min(H, int((i + 1) * ch + ch / 2))
            parches.append(img.crop((x0, y0, x1, y1)))

    feats = extraer_features(parches)               # [filas*cols, 768]
    probs = torch.softmax(clasificador(feats), dim=1)[:, idx_maligno]
    return probs.reshape(filas, cols)               # [filas, cols] en 0..1


def overlay_malignidad(imagen_pil, grid_probs, alpha=0.55, max_lado=512):
    """Superpone en ROJO las zonas que el modelo considera malignas.
    Las regiones benignas quedan casi sin teñir. Devuelve (imagen PIL, fraccion
    de la muestra marcada como maligna)."""
    img = imagen_pil.convert("RGB")
    if max(img.size) > max_lado:
        escala = max_lado / max(img.size)
        img = img.resize((int(img.width * escala), int(img.height * escala)))

    W, H = img.size
    m = grid_probs.numpy()
    m = Image.fromarray((m * 255).astype("uint8")).resize((W, H), Image.BICUBIC)
    radio = max(4, max(W, H) // 40)
    m = np.array(m.filter(ImageFilter.GaussianBlur(radio))) / 255.0

    base = np.array(img).astype(float) / 255.0
    rojo = np.zeros_like(base); rojo[..., 0] = 1.0
    a = alpha * m[..., None]
    over = (1 - a) * base + a * rojo
    frac = float((grid_probs > 0.5).float().mean())
    return Image.fromarray((np.clip(over, 0, 1) * 255).astype("uint8")), frac


def a_data_url(imagen_pil):
    """Convierte una imagen PIL a un data URL PNG (para mandar en JSON)."""
    buf = io.BytesIO()
    imagen_pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return "data:image/png;base64," + b64


class Clasificador(nn.Module):
    """Clasificador liviano que va encima de las features de Phikon."""

    def __init__(self, n_clases, dim_in=DIM_FEATURES):
        super().__init__()
        self.fc = nn.Linear(dim_in, n_clases)

    def forward(self, x):
        return self.fc(x)


def crear_clasificador(clases):
    return Clasificador(len(clases))


@torch.no_grad()
def similitud_a_prototipos(features, prototipos):
    """Similitud coseno de la imagen contra el prototipo (centro) de cada organo.
    features: tensor [1, 768] (CLS de Phikon).
    prototipos: dict {nombre_organo: tensor [768]}.
    Devuelve dict {nombre_organo: similitud_coseno (float)}."""
    f = torch.nn.functional.normalize(features, dim=1)[0]   # [768]
    salida = {}
    for organo, vec in prototipos.items():
        v = torch.nn.functional.normalize(vec.reshape(-1), dim=0)
        salida[organo] = float(torch.dot(f, v))
    return salida
