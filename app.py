# app.py
# Interfaz web para clasificar imagenes histologicas por benignidad,
# con soporte para varios organos (un clasificador por organo).
#
# Motor de IA: PHIKON (owkin/phikon), pre-entrenado en histopatologia,
# como extractor de features + un clasificador liviano por organo.
#
# Como usarla:
#   1) python -m pip install -r requirements.txt   (una sola vez)
#   2) python entrenar_benignidad.py Pulmon        (entrena un organo)
#   3) python app.py
#   4) Abri http://localhost:5000 en el navegador

import io
import os
import glob
from datetime import datetime

import torch
from PIL import Image
from flask import Flask, request, jsonify, render_template

import motor

app = Flask(__name__)
# Recargar las plantillas HTML al vuelo (sin reiniciar la app al editarlas).
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Diccionario de clasificadores cargados por organo:
# { "pulmon": {"clasificador":..., "clases":..., "organo":"Pulmon"} }
modelos = {}

# Prototipos de organo (para verificar que la imagen sea del organo elegido):
# { "Pulmon": tensor[768], "Colon": tensor[768] }  (los que tengan prototipo guardado)
prototipos = {}


def cargar_modelos():
    """Busca y carga todos los archivos 'modelo_<organo>.pth' del directorio.
    Devuelve la cantidad de modelos cargados."""
    modelos.clear()
    prototipos.clear()
    for ruta in glob.glob("modelo_*.pth"):
        try:
            datos = torch.load(ruta, weights_only=False)
            clases = datos["clases"]
            organo = datos.get("organo", "")
            if not organo:
                organo = os.path.basename(ruta)[len("modelo_"):-len(".pth")].capitalize()

            clasificador = motor.crear_clasificador(clases)
            clasificador.load_state_dict(datos["estado"])
            clasificador.eval()

            modelos[organo.lower()] = {
                "clasificador": clasificador,
                "clases": clases,
                "organo": organo,
                "parche_px": datos.get("parche_px"),   # escala si se entreno panoramica
            }
            # Prototipo de organo (si el modelo lo tiene guardado).
            proto = datos.get("prototipo")
            if proto is not None:
                prototipos[organo] = proto
        except Exception as e:
            print(f"[!] No se pudo cargar '{ruta}': {e}")
    return len(modelos)


def verificar_organo(organo_seleccionado, features):
    """Compara la imagen contra el prototipo de cada organo conocido.
    Devuelve un dict con el resultado de la verificacion, o None si no se puede
    verificar (no hay prototipos, o el organo elegido no tiene prototipo)."""
    if not prototipos or organo_seleccionado not in prototipos:
        return None

    sims = motor.similitud_a_prototipos(features, prototipos)   # {organo: cos}
    sim_sel = sims[organo_seleccionado]
    argmax = max(sims, key=sims.get)

    # Criterio SEGURO (mejor pedir confirmacion que dejar pasar un organo equivocado):
    # la imagen coincide si se parece CLARAMENTE al organo elegido, o si ese organo
    # es el mas parecido de todos (con un minimo).
    coincide = (sim_sel >= motor.UMBRAL_PROPIO) or \
               (argmax == organo_seleccionado and sim_sel >= motor.UMBRAL_ORGANO)

    # Para el aviso: si no coincide, vemos si se parece a OTRO organo (mismatch) o a
    # ninguno (no evaluable / OOD).
    otros = {o: v for o, v in sims.items() if o != organo_seleccionado}
    mejor_otro = max(otros, key=otros.get) if otros else None
    evaluable = coincide or (mejor_otro is not None and otros[mejor_otro] >= motor.UMBRAL_ORGANO)
    detectado = organo_seleccionado if coincide else (mejor_otro if evaluable else None)

    # Similitudes como % legibles (softmax con temperatura sobre las cosenos).
    import torch as _t
    vals = _t.tensor([sims[o] for o in sims]) / motor.TEMP_ORGANO
    probs = _t.softmax(vals, dim=0).tolist()
    similitudes = sorted(
        [{"organo": o, "valor": round(p * 100, 1)} for o, p in zip(sims.keys(), probs)],
        key=lambda x: x["valor"], reverse=True)

    return {
        "coincide": coincide,
        "evaluable": evaluable,
        "organo_seleccionado": organo_seleccionado,
        "organo_detectado": detectado if evaluable else None,
        "similitudes": similitudes,
    }


def lista_organos():
    return sorted(m["organo"] for m in modelos.values())


def _es_maligno(nombre):
    return "malig" in (nombre or "").lower()


def construir_informe(organo, probabilidades_ordenadas, frac_maligno, segunda, clases):
    """Arma un mini-informe de APOYO usando DOS modelos independientes:
    - el clasificador principal (Phikon) -> estimacion + %,
    - una segunda opinion (PLIP) benigno/maligno,
    y evalua si CONCUERDAN (lo mas informativo y honesto: dos modelos distintos
    de acuerdo refuerzan; si difieren, es un caso a revisar).
    NO describe morfologia en palabras porque en zero-shot no es confiable."""
    principal_clase = probabilidades_ordenadas[0]["clase"]
    conf = probabilidades_ordenadas[0]["valor"]

    # Nombres de clase para benigno / maligno segun las carpetas entrenadas.
    nombre_benigno = next((c for c in clases if "benig" in c.lower()), "Benigno")
    nombre_maligno = next((c for c in clases if "malig" in c.lower()), "Maligno")

    # Segunda opinion (PLIP): elegimos la clase con mayor probabilidad.
    seg_maligno = segunda["prob_maligno"] >= segunda["prob_benigno"]
    segunda_clase = nombre_maligno if seg_maligno else nombre_benigno
    segunda_valor = round((segunda["prob_maligno"] if seg_maligno
                           else segunda["prob_benigno"]) * 100, 1)

    # ¿Coinciden los dos modelos (en cuanto a benignidad)?
    concuerdan = _es_maligno(principal_clase) == seg_maligno
    if concuerdan:
        concordancia = (
            f"✅ Los dos modelos coinciden en «{principal_clase}». Son modelos "
            f"distintos e independientes, así que su acuerdo refuerza la estimación."
        )
    else:
        concordancia = (
            f"⚠️ Los modelos no coinciden — el de imagen estimó «{principal_clase}» "
            f"y la segunda opinión «{segunda_clase}». Conviene revisar este caso con "
            f"más atención."
        )

    # Dónde marcó cáncer: fracción de la muestra que el modelo considera maligna.
    pct = round(frac_maligno * 100)
    if frac_maligno < 0.05:
        donde = ("El mapa no marcó zonas malignas: el tejido se ve de aspecto benigno "
                 "en toda la muestra.")
    elif frac_maligno < 0.40:
        donde = (f"El mapa marcó en rojo zonas puntuales que considera malignas "
                 f"(~{pct}% de la muestra).")
    else:
        donde = (f"El mapa marcó en rojo amplias regiones que considera malignas "
                 f"(~{pct}% de la muestra).")

    return {
        "resumen": f"Estimación para {organo}: «{principal_clase}».",
        "principal": {"clase": principal_clase, "valor": conf, "modelo": "Phikon (imagen)"},
        "segunda": {"clase": segunda_clase, "valor": segunda_valor, "modelo": "PLIP (2ª opinión)"},
        "concuerdan": concuerdan,
        "concordancia": concordancia,
        "donde": donde,
        "limite": (
            "Informe orientativo generado por IA (dos modelos de apoyo). No es un "
            "diagnóstico ni un razonamiento médico: la validación final es del profesional."
        ),
    }


def predecir_imagen(organo, imagen_pil, forzar=False):
    """Corre Phikon + el clasificador del organo. Devuelve el dict de resultado,
    incluyendo un mapa de calor (donde miro la IA) y un texto explicativo.

    Antes de analizar, VERIFICA que la imagen sea del organo elegido. Si no coincide
    (o no parece una muestra valida) y 'forzar' es False, no analiza: devuelve un
    pedido de confirmacion para que el profesional decida."""
    entrada = modelos[organo]
    clasificador = entrada["clasificador"]
    clases = entrada["clases"]
    parche_px = entrada.get("parche_px")        # None = modelo clasico; nro = panoramico
    es_pano = parche_px is not None

    # Limitar el tamano de imagenes muy grandes (panoramicas / whole-slide) para no
    # agotar la memoria al cortarlas en muchos parches.
    MAX_LADO = 2048
    if max(imagen_pil.size) > MAX_LADO:
        escala = MAX_LADO / max(imagen_pil.size)
        nuevo = (int(imagen_pil.width * escala), int(imagen_pil.height * escala))
        imagen_pil = imagen_pil.resize(nuevo)
    idx_maligno = next((i for i, c in enumerate(clases) if "malig" in c.lower()),
                       len(clases) - 1)
    idx_benigno = next((i for i, c in enumerate(clases) if "benig" in c.lower()),
                       1 - idx_maligno if len(clases) == 2 else 0)

    # Feature de la imagen ENTERA. Se usa para verificar el organo (la identidad del
    # organo es una propiedad global, no de un parche) y, en modo clasico, clasificar.
    with torch.no_grad():
        feats_rep = motor.extraer_features(imagen_pil)

    # --- Verificacion de organo (antes de analizar) ---
    verif = verificar_organo(entrada["organo"], feats_rep)
    if verif and not verif["coincide"] and not forzar:
        return {"confirmacion_requerida": True, "verificacion": verif}

    # Mapa de MALIGNIDAD: clasifica region por region y marca en rojo lo maligno.
    grid = motor.mapa_malignidad(imagen_pil, clasificador, idx_maligno, tam=parche_px)
    overlay, frac_maligno = motor.overlay_malignidad(imagen_pil, grid)

    # Diagnostico general (headline):
    if es_pano:
        # En panoramico, la muestra es Maligna si hay al menos una region claramente
        # maligna; la confianza la da la region mas sospechosa.
        p_mal = float(grid.max())
        if p_mal >= 0.5:
            idx_diag, conf = idx_maligno, p_mal
        else:
            idx_diag, conf = idx_benigno, 1 - p_mal
        probs_clase = {idx_maligno: p_mal, idx_benigno: 1 - p_mal}
        diagnostico = clases[idx_diag]
        confianza = round(conf * 100, 2)
        probabilidades = [{"clase": c, "valor": round(probs_clase.get(i, 0.0) * 100, 2)}
                          for i, c in enumerate(clases)]
    else:
        with torch.no_grad():
            probabilidades_t = torch.nn.functional.softmax(clasificador(feats_rep)[0], dim=0)
        indice = probabilidades_t.argmax().item()
        diagnostico = clases[indice]
        confianza = round(probabilidades_t[indice].item() * 100, 2)
        probabilidades = [{"clase": c, "valor": round(probabilidades_t[i].item() * 100, 2)}
                          for i, c in enumerate(clases)]

    # Segunda opinion INDEPENDIENTE (PLIP): benigno / maligno.
    segunda = motor.segunda_opinion(imagen_pil, entrada["organo"])

    resultado = {
        "organo": entrada["organo"],
        "diagnostico": diagnostico,
        "confianza": confianza,
        "probabilidades": probabilidades,
        "mapa_calor": motor.a_data_url(overlay),
        "verificacion": verif,   # None si no se pudo verificar; util si se forzo
    }
    resultado["probabilidades"].sort(key=lambda x: x["valor"], reverse=True)
    resultado["informe"] = construir_informe(
        entrada["organo"], resultado["probabilidades"], frac_maligno, segunda, clases)
    return resultado


@app.route("/")
def inicio():
    return render_template("index.html", organos=lista_organos())


@app.route("/predecir", methods=["POST"])
def predecir():
    if not modelos and not cargar_modelos():
        return jsonify({
            "error": "Todavia no hay ningun modelo entrenado. "
                     "Corre 'python entrenar_benignidad.py <Organo>' primero."
        }), 400

    organo = request.form.get("organo", "").lower()
    if organo not in modelos:
        organo = sorted(modelos.keys())[0]

    if "imagen" not in request.files:
        return jsonify({"error": "No se recibio ninguna imagen."}), 400
    archivo = request.files["imagen"]
    if archivo.filename == "":
        return jsonify({"error": "No se selecciono ninguna imagen."}), 400

    try:
        imagen = Image.open(io.BytesIO(archivo.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "El archivo no es una imagen valida."}), 400

    # 'forzar': el profesional confirmo analizar aunque el organo no coincida.
    forzar = request.form.get("forzar", "").lower() in ("1", "true", "on", "si")

    return jsonify(predecir_imagen(organo, imagen, forzar=forzar))


@app.route("/feedback", methods=["POST"])
def feedback():
    """Guarda una imagen etiquetada por el profesional en la carpeta de su clase,
    para que se use en el proximo entrenamiento (human-in-the-loop)."""
    organo = request.form.get("organo", "").lower()
    if organo not in modelos:
        return jsonify({"error": "Organo invalido o sin modelo cargado."}), 400

    entrada = modelos[organo]
    clases = entrada["clases"]
    nombre_organo = entrada["organo"]

    etiqueta = request.form.get("etiqueta", "")
    if etiqueta not in clases:
        return jsonify({"error": f"Etiqueta invalida. Debe ser una de: {clases}"}), 400

    if "imagen" not in request.files or request.files["imagen"].filename == "":
        return jsonify({"error": "No se recibio ninguna imagen."}), 400

    archivo = request.files["imagen"]
    try:
        imagen = Image.open(io.BytesIO(archivo.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "El archivo no es una imagen valida."}), 400

    carpeta_destino = os.path.join("Datos", nombre_organo, etiqueta)
    os.makedirs(carpeta_destino, exist_ok=True)
    marca = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    nombre = f"feedback_{marca}.jpg"
    imagen.save(os.path.join(carpeta_destino, nombre), "JPEG", quality=95)

    total = len([f for f in os.listdir(carpeta_destino) if f.lower().endswith(".jpg")])
    return jsonify({
        "ok": True,
        "guardado_en": f"Datos/{nombre_organo}/{etiqueta}/{nombre}",
        "total_en_clase": total,
        "etiqueta": etiqueta,
    })


if __name__ == "__main__":
    n = cargar_modelos()
    if n:
        print(f"{n} modelo(s) cargado(s): {lista_organos()}")
        print("Cargando Phikon (motor de features)...")
        motor.cargar_backbone()
        print("Cargando PLIP (segunda opinión)...")
        motor.cargar_plip()
        print("Listo.")
    else:
        print("[!] No hay modelos entrenados todavia.")
        print("    Corre 'python entrenar_benignidad.py <Organo>' y reinicia la app.")

    print("\nAbri http://localhost:5000 en tu navegador.\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
