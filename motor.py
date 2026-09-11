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
from transformers import AutoImageProcessor, AutoModel

# Modelo base de patologia. ViT-B, devuelve vectores de 768 dimensiones.
MODELO_BASE = "owkin/phikon"
DIM_FEATURES = 768

# Se cargan una sola vez (son pesados) y se reutilizan.
_procesador = None
_backbone = None


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


@torch.no_grad()
def extraer_features(imagenes_pil):
    """Recibe una imagen PIL o una lista de imagenes PIL.
    Devuelve un tensor [N, 768] con las caracteristicas de Phikon (token CLS)."""
    procesador, backbone = cargar_backbone()
    if not isinstance(imagenes_pil, (list, tuple)):
        imagenes_pil = [imagenes_pil]
    entradas = procesador(imagenes_pil, return_tensors="pt")
    salida = backbone(**entradas)
    # El token CLS (posicion 0) resume toda la imagen.
    return salida.last_hidden_state[:, 0, :]


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
