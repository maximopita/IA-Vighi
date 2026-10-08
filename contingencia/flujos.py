"""Flujos de trabajo de cada estudio (mismos que el Excel y el sistema). Termina en INFORMADO."""

SECTORES = {
    "ingreso": "Ingreso",
    "laboratorio": "Laboratorio",
    "macroscopia": "Macroscopía",
    "traslados": "Traslados",
    "citotecnico": "Citotécnicos",
    "firmante": "Médicos firmantes",
}

# clave, nombre, sector que la realiza
BP = [
    ("macroscopia", "Macroscopía", "macroscopia"),
    ("procesamiento", "Procesamiento", "laboratorio"),
    ("inclusion", "Inclusión", "laboratorio"),
    ("corte", "Corte PH", "laboratorio"),
    ("coloreado", "Coloreado", "laboratorio"),
    ("microscopia", "Microscopía", "firmante"),
]
IHQ = [
    ("corte_ihq", "Corte IHQ", "laboratorio"),
    ("envio_ihq", "Envío lab. externo", "traslados"),
    ("proc_ihq", "Procesamiento IHQ", "laboratorio"),
    ("interp_ihq", "Interpretación IHQ", "firmante"),
]
PAP = [
    ("coloreado", "Coloreado", "laboratorio"),
    ("citotecnicos", "Citotécnicos", "traslados"),
    ("microscopia", "Microscopía", "firmante"),
]
CT = [
    ("coloreado", "Coloreado", "laboratorio"),
    ("microscopia", "Microscopía", "firmante"),
]
TIPOS = {"PAP": "PAP", "BP": "Biopsias", "CT": "Citologías"}
NOMBRE_ETAPA = {k: n for k, n, _ in BP + IHQ + PAP + CT}


def flujo(tipo, solicita_ihq=False):
    if tipo == "BP":
        return BP + (IHQ if solicita_ihq else [])
    return PAP if tipo == "PAP" else CT


def estado(tipo, hechas, solicita_ihq=False, anulado=False):
    """Devuelve (texto de estado, próxima etapa o None)."""
    if anulado:
        return "ANULADO", None
    for etapa in flujo(tipo, solicita_ihq):
        if etapa[0] not in hechas:
            return f"Pend. {etapa[1]}", etapa
    return ("INFORMADO (con IHQ)" if solicita_ihq else "INFORMADO"), None


def requisito(tipo, etapa, macro, micro, ihq):
    """Datos que tienen que estar cargados antes de marcar la etapa como lista (None = OK)."""
    if etapa == "macroscopia" and not (macro and (macro["descripcion"] or "").strip()):
        return "Falta la descripción macroscópica."
    if etapa == "microscopia":
        if tipo == "PAP" and not (micro and micro["bethesda"]):
            return "Falta el resultado (Bethesda)."
        if tipo != "PAP" and not (micro and ((micro["descripcion"] or "").strip() or (micro["conclusion"] or "").strip())):
            return "Falta el diagnóstico (descripción o conclusión)."
    if etapa == "interp_ihq" and not (ihq and (ihq["resultado"] or "").strip()):
        return "Falta el resultado de la IHQ."
    return None
