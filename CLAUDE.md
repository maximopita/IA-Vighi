# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

IA-Vighi is a local Flask app that classifies histology images as Benigno/Maligno per organ (Pulmón, Colon, Mama). It is decision-support only, not a validated diagnosis. Code, comments, UI and docs are in Spanish (identifiers and comments are written without accents/ñ); keep that style. `README.md` covers user-facing usage and `FUNDAMENTOS.md` the scientific references.

The repo root is the `IA-Vighi/` folder (the git repo). There is no test suite, linter or build step.

## Commands

```bash
python -m pip install -r requirements.txt        # once; torch is large, Windows needs LongPathsEnabled
python entrenar_benignidad.py <Organo>           # train one organ from Datos/<Organo>/<Clase>/ -> modelo_<organo>.pth
python entrenar_benignidad.py <Organo> --panoramica [--parche=256]   # cut wide images into patches
python app.py                                    # serves http://localhost:5000 (binds 0.0.0.0)
```

`Iniciar app.bat` launches the browser and `python app.py`. Run every script from the repo root: `Datos/` and `modelo_*.pth` are resolved relative to the current directory. Models are downloaded from Hugging Face on first run (Phikon ~335 MB, PLIP ~600 MB), then cached.

`Pruebas/<Organo>/{Benigno,Maligno,Mixtas}` hold sample images held out from training, for manual testing through the UI. `entrenar.py` is legacy (ResNet18 organ classifier reading `Organos/`), unused; don't extend it.

## Architecture

Two frozen pretrained models plus one small trained head per organ:

- **Phikon** (`owkin/phikon`, ViT, CLS token → 768-d) is the feature extractor. A single `nn.Linear(768, n_clases)` (`motor.Clasificador`) is trained on those features per organ. Only this head is saved in `modelo_<organo>.pth` (gitignored; must be regenerated with the training script, none are checked in).
- **PLIP** (`vinid/plip`) is an independent zero-shot second opinion, comparing the image to two fixed English prompts (`"benign {organ} tissue"` vs `"{organ} adenocarcinoma"`). Prompts were chosen empirically; organ names are translated ES→EN via `_ORGANO_EN` in `motor.py`, so a new organ needs an entry there or PLIP falls back to the Spanish name. Deliberately no free-text morphology descriptions (found unreliable in zero-shot; see README).

`motor.py` is the shared engine used by both `app.py` and `entrenar_benignidad.py`. It lazily loads the models as module-level singletons, and Phikon is loaded with `attn_implementation="eager"` so attention maps can be read.

### `.pth` contract (written by `entrenar_benignidad.py`, read by `app.cargar_modelos`)

`estado` (head weights), `clases` (sorted subfolder names), `organo`, `base`, `prototipo` (768-d organ centroid), `panoramica`, `parche_px`. The organ list in the UI is simply whichever `modelo_*.pth` files exist at startup, so adding an organ is: add `Datos/<Organo>/{Benigno,Maligno}/`, train, restart. Class handling in `app.py` finds classes by substring (`"malig"`, `"benig"`), so class folder names must contain those.

### Prediction flow (`app.predecir_imagen`)

1. Downscale to max 2048 px, extract Phikon features of the whole image.
2. **Organ verification**: cosine similarity to each organ's `prototipo` (thresholds `UMBRAL_PROPIO`/`UMBRAL_ORGANO` in `motor.py`). On mismatch or out-of-distribution, return `{"confirmacion_requerida": True}` unless `forzar` is sent; the UI then offers "Analizar igual". Prototypes are always computed from whole images, even in panoramic training.
3. **Malignancy map** (`motor.mapa_malignidad`): classify overlapping square windows (size `parche_px` for panoramic models, else 1/5 of the shorter side; at most `MAX_VENTANAS`) and average the probabilities per pixel. `overlay_malignidad` draws the p≥0.5 region as a red outline with a faint tint, and the p≥0.85 core in dark red. Cost is ~20 s on CPU for a large image; lower `MAX_VENTANAS` to trade detail for speed.
4. Headline diagnosis: for panoramic models, Maligno if any region ≥ 0.5 (confidence = worst region). For classic models, softmax of the head on whole-image features.
5. PLIP second opinion → `construir_informe` compares both and reports agreement/disagreement.

`motor.features_y_atencion` (attention rollout heatmap) exists but the current `/predecir` uses the malignancy map instead.

### Cellular engine (optional, nucleus by nucleus)

A second, independent engine marks each nucleus: HoVer-Net (`hovernet_fast-pannuke`, via TIAToolbox) in `motor_celular/nucleos.py`, classifying nuclei as Neoplásico / Inflamatorio / Conectivo / Muerto / Epitelial no neoplásico. It needs its own **Python 3.12 venv** (TIAToolbox does not support 3.14), so `app.py` launches it as a subprocess (`CELULAR_PY`, override with env var `VIGHI_CELULAR_PYTHON`; default `C:\vighi-venvs\celular\Scripts\python.exe`). The venv must live at a short path outside OneDrive: torch's file names exceed Windows' 260-char limit inside the long OneDrive path, and OneDrive would sync GBs. Setup: `py -3.12 -m venv C:\vighi-venvs\celular`, then `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`, then `pip install tiatoolbox`.

The UI button calls `POST /celular` (starts a background thread + process, returns a job id), polls `GET /celular/estado/<id>` (progress read from a JSON file the process updates), and `POST /celular/cancelar/<id>` kills it. Only one job runs at a time. If the venv is missing, the button renders disabled. It runs ~40 s for a 768 px tile and ~2 min for a 2048×1536 image on CPU (images are shrunk to 1536 px, background windows skipped). PanNuke weights are CC BY-NC-SA (non-commercial). On Mama benign samples with crowded nuclei it over-marks "Neoplásico"; treat it as visual support, not a diagnosis.

### Training modes

Classic mode treats each file as one pure tile of its class. `--panoramica` cuts each image into patches (`recortar_en_parches`, 25% overlap, background patches dropped by `_es_fondo`), and each patch inherits the folder's label. Training runs on precomputed features (full-batch, 60 epochs, 80/20 split with fixed seed), so feature extraction is the slow part.

### HTTP API (`templates/index.html` is a single-file UI)

- `GET /`: page, with the organ list rendered in.
- `POST /predecir`: form fields `imagen`, `organo`, optional `forzar`.
- `POST /feedback`: fields `imagen`, `etiqueta`, `organo`. This is the only place images are persisted, saved to `Datos/<Organo>/<Clase>/feedback_<timestamp>.jpg` for the next training round (human-in-the-loop). Plain predictions store nothing.

### UI styling

`templates/index.html` uses the institutional CAP Vighi palette, the same tokens as the NUEVAWEB site (`Escritorio/NUEVAWEB/susana-vighi-web/src/styles.css`, converted from oklch to hex): `--azul` #431866 (clinical-blue), `--acento` #a657ed (clinical-accent), `--gris` #665e77 (clinical-slate), surface #f8f5fb, radius 12px, Inter plus JetBrains Mono for eyebrow labels, and pill-shaped primary buttons. Keep the variable names and stay on these tokens when adding UI. Benigno green, Maligno red and amber warnings are deliberately not brand colors. Logos live in `static/` (`logo-blanco.png` in the header, copied from NUEVAWEB assets). Fonts load from Google Fonts and fall back to system fonts offline.

## Gotchas

- `README.md`'s last "Notas importantes" bullet (says the app doesn't detect a wrong organ) is stale: organ verification exists now.
- `app.cargar_modelos` calls `torch.load(..., weights_only=False)`; only load `.pth` files you trust.
- `Datos/` is tracked in git (thousands of images), and `*.pth` files are not.
