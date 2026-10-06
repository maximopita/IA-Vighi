@echo off
REM ============================================================
REM   Asistente de Benignidad (IA-Vighi)
REM   Doble clic para ARRANCAR la app. NO entrena: solo la usa.
REM   Los modelos ya entrenados se cargan solos.
REM   Si en esta PC faltan las librerias de IA, las instala la primera vez.
REM ============================================================
cd /d "%~dp0"
echo Iniciando el Asistente de Benignidad...
echo Python en uso:
python -c "import sys; print('  ' + sys.executable)"
echo.

REM Si falta torch (PC nueva), instala las dependencias con ESTE mismo Python.
python -c "import torch, transformers, flask" >nul 2>&1
if errorlevel 1 (
    echo Faltan las librerias de IA en esta PC. Instalando ^(puede tardar varios minutos^)...
    python -m pip install -r requirements.txt
    echo.
)

REM Aviso si no hay modelos entrenados (los .pth no se suben a git)
if not exist "modelo_*.pth" (
    echo [!] No hay modelos entrenados en esta carpeta ^(modelo_*.pth^).
    echo     Copialos desde la otra PC o entrena: python entrenar_benignidad.py Pulmon
    echo.
)

echo (La primera vez tarda unos segundos en cargar los modelos)
REM Abre el navegador recien cuando el servidor esta listo
start "" /b powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0esperar_y_abrir.ps1"
REM Arranca el servidor (dejar esta ventana abierta mientras uses la app)
python app.py
echo.
echo La app se detuvo. Si aparecio un error arriba, sacale captura. Podes cerrar esta ventana.
pause
