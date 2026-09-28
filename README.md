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
3. La app **verifica que la imagen sea del órgano elegido**. Si no coincide (o no
   parece una muestra válida), **avisa y pide confirmación** antes de analizar. Ver
   más abajo.
4. La IA estima **Benigno o Maligno** con un porcentaje de confianza.
5. Un **mini-informe de apoyo** combina **dos modelos independientes** y avisa si
   **concuerdan** (refuerza) o **difieren** (caso a revisar). Ver más abajo.
6. El profesional **confirma o corrige** el resultado, y esa imagen se guarda para
   mejorar el modelo en el próximo entrenamiento (*human-in-the-loop*).

Órganos disponibles actualmente: **Pulmón**, **Colon** y **Mama**.

### Verificación de órgano (que la foto sea del órgano elegido)

Antes de analizar, la app compara la imagen contra el "patrón" (centro) de cada órgano
en el espacio de características de Phikon:

- Si la imagen **se parece más a otro órgano** que al elegido → avisa
  *"Elegiste Pulmón, pero se parece más a Colon (97.8% vs 2.2%). ¿Analizar igual?"*.
- Si **no se parece a ninguna** muestra conocida (imagen rara o que no es histología)
  → avisa *"no parece una muestra válida"*.

En ambos casos **no bloquea**: el profesional decide con **"Analizar igual"** (analiza
de todas formas) o **"Elegir otra imagen"**. Se midió sobre las imágenes reales y la
separación Pulmón/Colon fue del **100%**. El "patrón" de cada órgano se calcula solo al
entrenar, así que un órgano nuevo queda cubierto automáticamente.

### El mini-informe de dos modelos

Para dar más respaldo, el resultado usa **dos modelos de IA distintos e
independientes**:

- **Phikon (modelo de imagen):** es el clasificador principal, entrenado con tus
  imágenes. Da la estimación Benigno/Maligno y el porcentaje.
- **PLIP (segunda opinión):** un modelo de patología distinto que compara la imagen
  contra descripciones de texto ("benign lung tissue" vs "lung adenocarcinoma") y
  estima, por su cuenta, Benigno/Maligno.

El informe muestra las dos opiniones y evalúa la **concordancia**:

- ✅ **Coinciden** → dos modelos distintos de acuerdo: refuerza la estimación.
- ⚠️ **Difieren** → es un caso más dudoso, conviene revisarlo con atención.

> **Importante — por qué NO describe la morfología en palabras:** se probó usar PLIP
> para escribir rasgos ("núcleos pleomórficos", "arquitectura cribriforme"…), pero en
> modo *zero-shot* esas descripciones finas **no son confiables** (se midió sobre las
> imágenes reales y fallaban seguido). Por eso el informe se limita a lo que **sí** es
> confiable: la estimación de cada modelo y su acuerdo/desacuerdo. Poner palabras
> médicas que el modelo no calcula bien sería engañoso y peligroso.

---

## Requisitos

- **Python 3.13** (o compatible)
- **Windows con rutas largas habilitadas** (ver más abajo, solo la primera vez)
- **Conexión a internet** la primera vez (para descargar los modelos: Phikon ~335 MB
  y PLIP ~600 MB; después quedan en caché y funciona offline)

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

### Entrenar desde imágenes PANORÁMICAS (detección de zonas)

Hay dos formas de entrenar:

- **Recortes puros (clásico):** cada imagen de `Datos/<Organo>/<Clase>/` es toda de
  esa clase (un tile benigno o uno maligno). Es lo que usan pulmón, colon y mama hoy.
- **Panorámicas (`--panoramica`):** ponés **imágenes amplias** (campos grandes). El
  script **las corta solo en parches**, descarta el fondo (vidrio) y entrena con esos
  parches. Así el modelo aprende a la **misma escala** en que después analiza, y marca
  mejor las **zonas** de cáncer dentro de una imagen grande.

```bash
python entrenar_benignidad.py <Organo> --panoramica
python entrenar_benignidad.py <Organo> --panoramica --parche=256   # tamaño de parche
```

> **Importante:** en modo panorámica, cada imagen que pongas en `Benigno/` debe ser un
> **campo predominantemente sano**, y cada una en `Maligno/` un **campo con tumor**. Si
> una panorámica mezcla mucho sano y tumor, conviene recortar antes la zona de cada
> clase (o usar el modo clásico con recortes). La **escala** (aumento) de las imágenes
> de entrenamiento debería ser parecida a la de las que vas a analizar.

Al analizar, la app detecta que el modelo es panorámico (guarda el tamaño de parche) y
recorre la imagen grande a esa misma escala para marcar las zonas.

---

## Estructura del proyecto

| Archivo / carpeta | Qué es |
|---|---|
| `app.py` | La aplicación web (interfaz + predicción + feedback) |
| `motor.py` | Los motores de IA: Phikon (clasificación) y PLIP (segunda opinión) |
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
