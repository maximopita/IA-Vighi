# Espera a que el servidor de la app responda y recien ahi abre el navegador.
# Lo llama "Iniciar app.bat". Espera hasta ~3 minutos.
for ($i = 0; $i -lt 90; $i++) {
    try {
        Invoke-WebRequest "http://localhost:5000" -UseBasicParsing -TimeoutSec 2 | Out-Null
        Start-Process "http://localhost:5000"
        break
    } catch {
        Start-Sleep -Seconds 2
    }
}
