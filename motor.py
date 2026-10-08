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
# de features de Phikon, por similitud coseno. Las similitudes coseno de Phikon son
# TODAS altas (todo tejido histologico se parece), asi que un umbral ABSOLUTO deja
# pasar el organo equivocado. Lo confiable es el ORDEN: el organo correcto es el mas
# parecido. Por eso el criterio es RELATIVO (ver app.verificar_organo).
UMBRAL_ORGANO = 0.45   # piso de similitud: por debajo, no hay ningun organo parecido (OOD)
MARGEN_ORGANO = 0.06   # el elegido coincide si esta a <= este margen del mas parecido
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


MAX_VENTANAS = 144          # tope de ventanas en imagenes grandes (controla el tiempo en CPU)
MAX_VENTANAS_CHICA = 64     # tope para imagenes chicas (< 1200 px): no necesitan tanto detalle
# Elegidos midiendo sobre Pruebas/Colon (6 benignas + 6 malignas): con ventana 1/4 y
# umbral 0.7 las benignas marcaron 0.9% (max 5%) y las malignas 99.9%. Con ventana 1/5
# y umbral 0.5 las benignas marcaban 10.9% (max 38%): mas detalle = mas falsos positivos.
UMBRAL_MALIGNO = 0.7  # probabilidad desde la cual una zona se marca como sospechosa
UMBRAL_NUCLEO = 0.9   # zona de alta confianza (se marca mas fuerte)


def _posiciones(largo, ventana, paso):
    """Posiciones de inicio de las ventanas a lo largo de un eje, cubriendo el borde."""
    if largo <= ventana:
        return [0]
    pos = list(range(0, largo - ventana + 1, paso))
    if pos[-1] != largo - ventana:
        pos.append(largo - ventana)
    return pos


@torch.no_grad()
def mapa_malignidad(imagen_pil, clasificador, idx_maligno, tam=None, max_lado=512,
                    divisor=4, temperatura=1.0):
    """Mapa de malignidad de alta resolucion, por ventanas SUPERPUESTAS.
    Desliza una ventana cuadrada por la imagen (con ~2/3 de solapamiento), clasifica
    cada una como benigna/maligna y promedia las probabilidades en cada pixel
    (cada ventana pesa un poco menos en su borde). Asi el mapa tiene detalle fino, sin el
    aspecto de 'casilleros' de una grilla.
    A diferencia del mapa de atencion ('donde miro'), esto muestra 'donde considera
    que hay cancer'.

    'tam' (px) es la escala con que se entreno un modelo panoramico; si no se pasa,
    la ventana mide 1/'divisor' del lado menor de la imagen.
    Devuelve (mapa, probs_ventanas):
      mapa: tensor [Hs, Ws] en 0..1 (Hs, Ws = tamano de trabajo, lado mayor <= max_lado)
      probs_ventanas: tensor [N] con la prob. de MALIGNO de cada ventana."""
    img = imagen_pil.convert("RGB")
    W, H = img.size
    ventana = int(tam) if tam else max(64, round(min(W, H) / divisor))
    ventana = min(ventana, W, H)
    paso = max(8, ventana // 3)
    # Si saldrian demasiadas ventanas, agrandamos el paso hasta cumplir el tope.
    tope = MAX_VENTANAS if max(W, H) > 1200 else MAX_VENTANAS_CHICA
    while len(_posiciones(W, ventana, paso)) * len(_posiciones(H, ventana, paso)) > tope:
        paso = int(paso * 1.15) + 1
    xs, ys = _posiciones(W, ventana, paso), _posiciones(H, ventana, paso)
    coords = [(x, y) for y in ys for x in xs]

    feats = extraer_features([img.crop((x, y, x + ventana, y + ventana)) for x, y in coords])
    # temperatura: >1 suaviza la confianza (calibracion). =1 no cambia nada.
    probs = torch.softmax(clasificador(feats) / temperatura, dim=1)[:, idx_maligno]

    # Acumulamos en un lienzo reducido (lado mayor = max_lado) con peso casi plano.
    s = min(1.0, max_lado / max(W, H))
    Ws, Hs = max(1, round(W * s)), max(1, round(H * s))
    v = max(2, round(ventana * s))
    # Peso casi plano (borde suave): mantiene el detalle fino, sin redondear las zonas.
    borde = max(1, v // 4)
    perfil = np.ones(v); rampa = np.linspace(0.2, 1, borde, endpoint=False)
    perfil[:borde] = rampa; perfil[-borde:] = rampa[::-1]
    peso_v = np.outer(perfil, perfil)
    acum = np.zeros((Hs, Ws)); peso = np.zeros((Hs, Ws))
    for (x, y), p in zip(coords, probs.tolist()):
        x0, y0 = int(round(x * s)), int(round(y * s))
        x1, y1 = min(Ws, x0 + v), min(Hs, y0 + v)
        w = peso_v[:y1 - y0, :x1 - x0]
        acum[y0:y1, x0:x1] += p * w
        peso[y0:y1, x0:x1] += w
    mapa = torch.from_numpy(acum / np.maximum(peso, 1e-6)).float()
    return mapa, probs


def _contorno(mascara_bool, grosor=2):
    """Borde de una mascara booleana (numpy), de 'grosor' px aprox."""
    m = Image.fromarray((mascara_bool * 255).astype("uint8"))
    k = 2 * grosor + 1
    interior = np.array(m.filter(ImageFilter.MinFilter(k))) > 0
    return mascara_bool & ~interior


def overlay_malignidad(imagen_pil, mapa, max_lado=512):
    """Marca las zonas malignas con CONTORNOS rojos y un tinte suave adentro, en vez
    de una mancha difusa. Doble nivel: el contorno fino delimita lo sospechoso
    (prob >= UMBRAL_MALIGNO) y el interior mas intenso, lo de alta confianza
    (>= UMBRAL_NUCLEO).
    Lo benigno queda sin tocar. Devuelve (imagen PIL, fraccion marcada como maligna)."""
    Hs, Ws = mapa.shape
    img = imagen_pil.convert("RGB").resize((Ws, Hs), Image.LANCZOS)
    m = mapa.numpy()

    # Limpieza de la mascara: se descartan manchitas y se suaviza el borde escalonado.
    def limpiar(mask):
        im = Image.fromarray((mask * 255).astype("uint8"))
        r = max(1, max(Ws, Hs) // 170)
        im = im.filter(ImageFilter.MinFilter(2 * r + 1)).filter(ImageFilter.MaxFilter(2 * r + 1))
        im = im.filter(ImageFilter.ModeFilter(2 * r + 1))
        return np.array(im) > 0

    sospechoso = limpiar(m >= UMBRAL_MALIGNO)
    nucleo = limpiar(m >= UMBRAL_NUCLEO) & sospechoso

    base = np.array(img).astype(float) / 255.0
    rojo = np.array([0.90, 0.10, 0.12])
    tinte = np.where(nucleo, 0.34, np.where(sospechoso, 0.16, 0.0))[..., None]
    over = (1 - tinte) * base + tinte * rojo

    grosor = max(1, max(Ws, Hs) // 260)
    borde = _contorno(sospechoso, grosor)
    over[borde] = rojo
    borde_n = _contorno(nucleo, grosor) & ~borde
    over[borde_n] = np.array([0.55, 0.0, 0.10])   # rojo oscuro: nucleo de alta confianza

    frac = float(sospechoso.mean())
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
    """Similitud coseno de la imagen contra el/los prototipo(s) de cada organo.
    features: tensor [1, 768] (CLS de Phikon).
    prototipos: dict {nombre_organo: tensor [768]  O  lista de tensores [768]}.
    Cada organo puede tener VARIOS prototipos (p. ej. uno 'chico'/tile y otro
    'ancho'/campo amplio); se devuelve la MEJOR coincidencia. Asi reconoce el
    organo tanto en recortes como en imagenes panoramicas.
    Devuelve dict {nombre_organo: similitud_coseno (float)}."""
    f = torch.nn.functional.normalize(features, dim=1)[0]   # [768]
    salida = {}
    for organo, protos in prototipos.items():
        if not isinstance(protos, (list, tuple)):
            protos = [protos]
        sims = []
        for vec in protos:
            v = torch.nn.functional.normalize(vec.reshape(-1), dim=0)
            sims.append(float(torch.dot(f, v)))
        salida[organo] = max(sims) if sims else 0.0
    return salida
