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


def cargar_modelos():
    """Busca y carga todos los archivos 'modelo_<organo>.pth' del directorio.
    Devuelve la cantidad de modelos cargados."""
    modelos.clear()
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
            }
        except Exception as e:
            print(f"[!] No se pudo cargar '{ruta}': {e}")
    return len(modelos)


def lista_organos():
    return sorted(m["organo"] for m in modelos.values())


def construir_explicacion(organo, probabilidades_ordenadas, mask):
    """Genera un texto honesto que explica COMO la IA llego al resultado.
    No inventa un razonamiento medico: describe que tan decidida fue la
    estimacion y donde se concentro la atencion."""
    diag = probabilidades_ordenadas[0]["clase"]
    conf = probabilidades_ordenadas[0]["valor"]
    segundo = probabilidades_ordenadas[1]["valor"] if len(probabilidades_ordenadas) > 1 else 0.0
    margen = round(conf - segundo, 1)

    # Que tan decidida fue la estimacion (segun la diferencia entre las dos opciones).
    if margen >= 90:
        decision = "Es una estimación muy decidida"
    elif margen >= 40:
        decision = "Es una estimación decidida"
    else:
        decision = ("Es una estimación poco decidida: conviene tomarla como un caso "
                    "dudoso y revisarlo con atención")

    # Que tan concentrada estuvo la atencion (fraccion de la imagen 'caliente').
    frac = (mask > 0.5).float().mean().item()
    if frac < 0.15:
        foco = "se concentró en zonas puntuales del tejido"
    elif frac < 0.40:
        foco = "se concentró en varias regiones del tejido"
    else:
        foco = "distribuyó su atención en buena parte de la muestra"

    return (
        f"La IA estimó «{diag}» para {organo} con {conf}% de confianza. "
        f"{decision} (la diferencia con la otra opción es de {margen} puntos). "
        f"Para decidir, comparó las características de esta imagen con los patrones que "
        f"aprendió de muestras de {organo} ya clasificadas por especialistas; según el "
        f"mapa de atención, {foco} (zonas en rojo). "
        f"⚠️ Este texto explica cómo la IA llegó al resultado — no es un razonamiento "
        f"diagnóstico médico. La validación final es del profesional."
    )


def predecir_imagen(organo, imagen_pil):
    """Corre Phikon + el clasificador del organo. Devuelve el dict de resultado,
    incluyendo un mapa de calor (donde miro la IA) y un texto explicativo."""
    entrada = modelos[organo]
    clasificador = entrada["clasificador"]
    clases = entrada["clases"]

    with torch.no_grad():
        features, mask = motor.features_y_atencion(imagen_pil)   # [1,768], mapa
        salida = clasificador(features)
        probabilidades = torch.nn.functional.softmax(salida[0], dim=0)

    indice = probabilidades.argmax().item()

    # Mapa de calor superpuesto (attention rollout).
    overlay = motor.overlay_mapa(imagen_pil, mask)

    resultado = {
        "organo": entrada["organo"],
        "diagnostico": clases[indice],
        "confianza": round(probabilidades[indice].item() * 100, 2),
        "probabilidades": [
            {"clase": clase, "valor": round(probabilidades[i].item() * 100, 2)}
            for i, clase in enumerate(clases)
        ],
        "mapa_calor": motor.a_data_url(overlay),
    }
    resultado["probabilidades"].sort(key=lambda x: x["valor"], reverse=True)
    resultado["explicacion"] = construir_explicacion(
        entrada["organo"], resultado["probabilidades"], mask)
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

    return jsonify(predecir_imagen(organo, imagen))


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
        print("Listo.")
    else:
        print("[!] No hay modelos entrenados todavia.")
        print("    Corre 'python entrenar_benignidad.py <Organo>' y reinicia la app.")

    print("\nAbri http://localhost:5000 en tu navegador.\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
