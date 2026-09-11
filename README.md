# 🔬 Asistente de Benignidad (IA-Vighi)

Clasificador de imágenes histológicas que estima **Benigno / Maligno** por órgano,
usando inteligencia artificial.

El "motor" de IA es **Phikon** (`owkin/phikon`), un modelo pre-entrenado
específicamente en **histopatología**, que extrae las características de cada imagen.
Encima va un clasificador liviano que aprende a distinguir Benigno / Maligno para
cada órgano.

Es una herramienta de **apoyo al diagnóstico** pensada para agilizar el trabajo del
patólogo — **no reemplaza el diagnóstico profesional**. La validación final siempre
es del especialista.

Corre de forma **local** en tu máquina (solo necesita internet la primera vez, para
descargar el modelo Phikon).

> ⚖️ **Licencia de Phikon:** "Owkin non-commercial license". Uso permitido para
> **investigación e interno**, NO comercial. Para un producto comercial habría que
> licenciar con Owkin o usar otro motor.

---

## ¿Qué hace?

1. Elegís el **órgano** de la muestra en la interfaz web.
2. Arrastrás una **imagen** histológica.
3. La IA estima **Benigno o Maligno** con un porcentaje de confianza.
4. El profesional **confirma o corrige** el resultado, y esa imagen se guarda para
   mejorar el modelo en el próximo entrenamiento (*human-in-the-loop*).

Órganos disponibles actualmente: **Pulmón** y **Colon**.

---

## Requisitos

- **Python 3.13** (o compatible)
- **Windows con rutas largas habilitadas** (ver más abajo, solo la primera vez)
- **Conexión a internet** la primera vez (para descargar Phikon, ~335 MB; después
  queda en caché y funciona offline)

---

## Instalación (una sola vez)

### 1. Habilitar rutas largas en Windows

PyTorch necesita esto para instalarse. Abrí **PowerShell como administrador** y corré:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

### 2. Instalar las dependencias

En una terminal normal, dentro de la carpeta del proyecto:

```bash
python -m pip install -r requirements.txt
```

Instala PyTorch, torchvision, Flask, Pillow y transformers (Hugging Face).
La descarga es grande, puede tardar unos minutos.

---

## Uso

### Levantar la interfaz

```bash
python app.py
```

Al arrancar carga Phikon (unos segundos). Después abrí **http://localhost:5000**.

- Elegí el **órgano** en el selector.
- Arrastrá una imagen (o hacé clic para elegirla).
- Mirá el resultado (Benigno/Maligno + porcentajes).
- Debajo del resultado, el **mapa de calor** ("¿Dónde miró la IA?") muestra en rojo
  las zonas que más pesaron en la decisión (attention rollout). Es orientativo: indica
  *dónde* se concentró la IA, no una explicación médica en palabras.
- Opcional: usá los botones **✓ Concuerdo** / **✗ No, es...** para dar feedback.

Para frenar la app: `Ctrl + C` en la terminal.

### ¿Se guardan las imágenes que subo?

- **Solo predecir** (arrastrar y ver el resultado): **NO** se guarda nada, la imagen
  queda solo en memoria.
- **Usar los botones de feedback** (✓ / ✗): **SÍ** se guarda la imagen en
  `Datos/<Organo>/<Clase>/`, para el próximo entrenamiento.

---

## Entrenar un modelo

El modelo aprende de las imágenes que pongas en `Datos/<Organo>/`.

### Estructura de carpetas

```
Datos/
  Pulmon/
    Benigno/     ← imágenes benignas de pulmón
    Maligno/     ← imágenes malignas de pulmón
  Colon/
    Benigno/
    Maligno/
```

### Comando de entrenamiento

```bash
python entrenar_benignidad.py Pulmon
```

(cambiá `Pulmon` por el órgano que quieras). El script:

1. Extrae las características de cada imagen con Phikon (esto tarda unos minutos en
   CPU — es la parte lenta, pero se hace una sola vez).
2. Entrena el clasificador liviano encima (rápido).
3. Guarda `modelo_<organo>.pth`.

Reiniciá la app y el órgano aparece solo en el selector.

**Recomendación de imágenes:** al menos 20–30 por clase para probar; 200+ por clase
para resultados confiables. Mantené las clases **balanceadas**.

### Agregar un órgano nuevo

1. Creá `Datos/<NuevoOrgano>/Benigno/` y `Datos/<NuevoOrgano>/Maligno/` con imágenes.
2. Corré `python entrenar_benignidad.py <NuevoOrgano>`.
3. Reiniciá `python app.py`.

---

## Estructura del proyecto

| Archivo / carpeta | Qué es |
|---|---|
| `app.py` | La aplicación web (interfaz + predicción + feedback) |
| `motor.py` | El motor de IA: carga Phikon y extrae características |
| `entrenar_benignidad.py` | Entrena un clasificador de benignidad por órgano |
| `templates/index.html` | La interfaz visual |
| `requirements.txt` | Las dependencias de Python |
| `modelo_<organo>.pth` | Clasificador entrenado de cada órgano (se genera al entrenar) |
| `Datos/<Organo>/<Clase>/` | Imágenes de entrenamiento |
| `Pruebas/<Organo>/<Clase>/` | Imágenes de prueba (no usadas en el entrenamiento) |
| `entrenar.py` | (Antiguo) clasificador de órgano completo, en desuso |

> Los archivos `.pth` guardan solo el clasificador liviano (unos KB). El motor pesado
> (Phikon) se descarga aparte y queda en la caché de Hugging Face. No se suben a git.

---

## Notas importantes

- **No es un diagnóstico médico validado.** Es un apoyo para agilizar el triage. La
  decisión clínica es siempre del profesional.
- Los porcentajes indican **grado de parecido** con lo que la IA aprendió, no una
  probabilidad clínica.
- El modelo **solo verifica benignidad dentro del órgano que elegís**: no detecta si
  la imagen es de otro órgano. Si elegís "Pulmón" y subís un colon, igual va a
  responder Benigno/Maligno (no avisa la discordancia).
- La precisión depende de la calidad y cantidad de las imágenes de entrenamiento.
