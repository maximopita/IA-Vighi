@echo off
REM ============================================================
REM   Asistente de Benignidad (IA-Vighi)
REM   Doble clic para ARRANCAR la app. NO entrena: solo la usa.
REM   Los modelos ya entrenados se cargan solos.
REM ============================================================
cd /d "%~dp0"
echo Iniciando el Asistente de Benignidad...
echo (La primera vez tarda unos segundos en cargar los modelos)
echo.
REM Abre el navegador en la interfaz
start "" http://localhost:5000
REM Arranca el servidor (dejar esta ventana abierta mientras uses la app)
python app.py
echo.
echo La app se detuvo. Podes cerrar esta ventana.
pause
