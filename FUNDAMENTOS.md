# Fundamentación científica y fuentes

Este documento respalda **por qué** cada parte del proyecto es posible y **de dónde**
sacar más información. Pensado para poder justificar el enfoque ante colegas, o para
profundizar.

> Aclaración transversal: esto es una herramienta de **apoyo/triage**, no un
> diagnóstico validado. La literatura de abajo respalda la *viabilidad técnica*, no
> la validación clínica de *este* sistema en particular (ver sección final).

---

## 1. Clasificar histología con IA (redes neuronales sobre imágenes)

**Qué hacemos:** una red neuronal analiza la imagen y estima una categoría.

**Por qué se puede:** las redes convolucionales (CNN) y luego los *Vision
Transformers* (ViT) demostraron rendimiento a nivel experto en muchas tareas de
imagen, incluida la imagen médica.

**Fuentes:**
- He et al., *Deep Residual Learning for Image Recognition* (ResNet), CVPR 2016 —
  arXiv:1512.03385. (La arquitectura ResNet que usamos al principio.)
- Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image
  Recognition at Scale* (ViT), ICLR 2021 — arXiv:2010.11929. (La arquitectura del
  motor actual, Phikon, es un ViT.)

---

## 2. Transfer learning y "foundation models" de patología (Phikon)

**Qué hacemos:** no entrenamos la red desde cero; partimos de un modelo pre-entrenado
en histopatología (Phikon) y le agregamos un clasificador liviano.

**Por qué se puede:** un modelo entrenado con millones de imágenes de tejido aprende
representaciones generales (núcleos, glándulas, texturas) que después sirven para
tareas nuevas con pocos datos. Phikon fue entrenado con **~460 millones de "tiles"**
de patología de **+100 cohortes públicas** y **>30 tipos de cáncer**, con aprendizaje
auto-supervisado.

**Fuentes:**
- Filiot et al., *Scaling Self-Supervised Learning for Histopathology with Masked
  Image Modeling*, medRxiv 2023. Código y modelo: GitHub `owkin/HistoSSLscaling`;
  modelo en Hugging Face `owkin/phikon`.
- Filiot et al., *Phikon-v2: A large and public feature extractor for biomarker
  prediction*, 2024 — arXiv:2409.09173.
- Panorama general de foundation models en patología: Chen et al., *Towards a
  general-purpose foundation model for computational pathology* (UNI), Nature
  Medicine, 2024.

**Licencia (importante para uso):** Phikon usa la *Owkin non-commercial license* —
válido para investigación/interno, no comercial.

---

## 3. El dataset de prueba (LC25000)

**Qué hacemos:** para el prototipo entrenamos con imágenes públicas de pulmón y colon,
etiquetadas como benigno / adenocarcinoma / carcinoma escamoso.

**Fuente:**
- Borkowski et al., *Lung and Colon Cancer Histopathological Image Dataset
  (LC25000)*, 2019 — arXiv:1912.12142. 25.000 imágenes, 5 clases, 768×768,
  de-identificadas y de libre acceso para investigación.

**Nota:** es un dataset "limpio y parejo". Por eso da precisiones muy altas
(~100%). Con imágenes reales del laboratorio la precisión será menor: es esperable.

---

## 4. El mapa de calor ("¿dónde miró la IA?")

**Qué hacemos:** mostramos en rojo las zonas de la imagen que más influyeron en la
decisión.

**Por qué se puede:** hay métodos post-hoc de "IA explicable" (XAI) que estiman la
importancia de cada región:
- Para los ViT (nuestro caso) usamos **attention rollout**: combina las matrices de
  atención de todas las capas para estimar cuánto atiende el modelo a cada parche.
- La primera versión probó **Grad-CAM** (basado en gradientes), que en ViT dio un
  mapa muy concentrado; por eso pasamos a attention rollout.

**Fuentes:**
- Abnar & Zuidema, *Quantifying Attention Flow in Transformers* (attention rollout),
  ACL 2020 — aclanthology.org/2020.acl-main.385.
- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization*, ICCV 2017 — arXiv:1610.02391.

**Límite honesto (hay debate científico):** un mapa de atención muestra *dónde* se
concentró el modelo, no necesariamente el "porqué" causal. Conviene conocer el
debate:
- Jain & Wallace, *Attention is not Explanation*, NAACL 2019.
- Wiegreffe & Pinter, *Attention is not not Explanation*, EMNLP 2019.
- Rudin, *Stop explaining black box machine learning models for high stakes
  decisions and use interpretable models instead*, Nature Machine Intelligence 2019.

Por eso en la interfaz decimos "muestra dónde miró, no una explicación médica".

---

## 5. Dónde ampliar / mantenerse al día

- **Modelos de patología (Hugging Face):** buscar `owkin/phikon`, `MahmoodLab/UNI`,
  `prov-gigapath`, etc. (cada uno con su model card y licencia).
- **Repos de referencia:** `owkin/HistoSSLscaling` (Phikon), `mahmoodlab/UNI`.
- **Revisiones (para contexto general):** buscar "computational pathology foundation
  models review" en PubMed / Nature Medicine / Nature Reviews.
- **Reproducibilidad de datasets:** The Cancer Genome Atlas (TCGA), sobre el que se
  entrenaron varios de estos modelos.

---

## 6. Lo que la literatura NO cubre (y hay que hacer)

Para pasar de prototipo a uso real, la fundamentación técnica no alcanza; hace falta:

1. **Validar con datos propios** del laboratorio, etiquetados por patólogos, y medir
   sensibilidad/especificidad en casos que la IA nunca vio.
2. **Marco regulatorio.** Un software de apoyo al diagnóstico suele ser un "producto
   médico / software as a medical device":
   - Argentina: **ANMAT**.
   - Referencias internacionales: **FDA** (Software as a Medical Device, AI/ML),
     **EU MDR**, guías **IMDRF**.
3. **Documentar limitaciones y sesgos** (un modelo puede fallar con tinciones,
   escáneres o poblaciones distintas a las de entrenamiento — ver también literatura
   sobre robustez entre centros médicos).

---

_Las referencias son puntos de partida; verificá siempre la versión y el enlace
actual antes de citarlas formalmente._
