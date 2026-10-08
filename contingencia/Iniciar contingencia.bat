@echo off
title Sistema de contingencia CAP Vighi
cd /d "%~dp0"
rem Carpeta extra para respaldos (ej. una biblioteca de SharePoint sincronizada). Dejar vacio si no se usa.
set CONTINGENCIA_RESPALDO=
py -m pip install -q -r requirements.txt
py app.py
pause
