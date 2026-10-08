# Sistema web de contingencia — CAP Vighi

Sistema provisorio para cuando southernbits no está disponible. Corre en **una PC del laboratorio** (servidor)
y el resto entra con el navegador por la **red interna**: funciona aunque no haya internet.

## Poner en marcha

1. En la PC servidor (con Python 3 instalado), doble clic en **`Iniciar contingencia.bat`**.
   La ventana muestra las direcciones, por ejemplo `http://192.168.1.50:8000`. **No cerrar esa ventana.**
2. **Primera vez:** desde la PC servidor abrir `http://localhost:8000` y crear el usuario administrador.
3. En **Usuarios**, darle a cada persona una clave temporal (al entrar la cambia por una propia).
   Los usuarios y sectores ya vienen cargados desde el Excel; revisar nombres y sectores.
4. El resto de las PCs entra con la dirección que muestra la ventana.

Si Windows pregunta por el firewall, permitir el acceso en **redes privadas**.

## Uso

- **Ingreso:** `+ PAP`, `+ Biopsia`, `+ Citología`. El lote de biopsias se arma solo (`ONCO-0930.1`).
- **Tablero:** pendientes, informados, sin cargar al sistema; filtros por estudio y sector; búsqueda.
- **Ficha del caso:** recorrido de etapas con el botón **✔ Listo** (queda registrado quién y cuándo),
  macroscopía y microscopía con **templates** (el texto aparece editable), IHQ, y "Vuelta al sistema".
- **Deshacer:** la última etapa la puede deshacer quien la registró o un administrador.
- **Exportar Excel:** todos los casos con sus etapas, para re-cargarlos en southernbits.

## Datos y respaldos

- Base de datos: `data/contingencia.db` (SQLite). **Contiene datos de pacientes.**
- Respaldo automático cada 10 minutos en `data/respaldos/` (se guardan los últimos 48).
- Respaldo extra en otra carpeta (ej. biblioteca de SharePoint sincronizada): completar
  `CONTINGENCIA_RESPALDO` en `Iniciar contingencia.bat`.
- Cuando todo esté re-cargado en el sistema, archivar o borrar la base según la política del laboratorio.

## Configuración (variables de entorno)

| Variable | Para qué | Por defecto |
|---|---|---|
| `CONTINGENCIA_PUERTO` | Puerto web | `8000` |
| `CONTINGENCIA_DB` | Ruta de la base | `data/contingencia.db` |
| `CONTINGENCIA_RESPALDO` | Carpeta extra de respaldos | — |
| `CONTINGENCIA_ESTRICTO` | `1` = cada etapa solo la marca su sector | `0` |

## Actualizar templates / catálogo / usuarios

`py preparar_semillas.py` regenera `seed/*.json` desde el Excel y los CSV. Las semillas se cargan
solo en una base nueva (vacía).

## Estructura

- `app.py` — pantallas y reglas · `flujos.py` — etapas de PAP / BP (+IHQ) / CT
- `db.py` — **todo el acceso a datos** (SQL estándar; para migrar a MySQL se cambia solo este módulo)
- `templates/`, `static/` — interfaz · `seed/` — datos iniciales
