"""Acceso a datos. Todo el SQL de la app pasa por acá (SQL estándar) para poder migrar a MySQL
cambiando solo este módulo: conexión, placeholders (? -> %s) y los tipos del esquema."""
import json
import os
import shutil
import sqlite3
import threading
import time
from datetime import datetime

from flask import g

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
DB_PATH = os.environ.get("CONTINGENCIA_DB", os.path.join(DATA, "contingencia.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS usuarios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iniciales VARCHAR(10) NOT NULL UNIQUE,
    nombre VARCHAR(100),
    sectores VARCHAR(200) NOT NULL DEFAULT '',
    admin INTEGER NOT NULL DEFAULT 0,
    activo INTEGER NOT NULL DEFAULT 1,
    clave_hash VARCHAR(255),
    debe_cambiar_clave INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS casos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tipo VARCHAR(3) NOT NULL,
    protocolo VARCHAR(30) NOT NULL,
    lote VARCHAR(40), tipo_lote VARCHAR(20), n_lote INTEGER,
    fecha_recoleccion VARCHAR(10),
    dni VARCHAR(20), cobertura VARCHAR(40), n_afiliado VARCHAR(40),
    nombre VARCHAR(80), apellido VARCHAR(80), sexo VARCHAR(10), exento VARCHAR(3),
    fecha_nacimiento VARCHAR(10), email VARCHAR(120), telefono VARCHAR(40),
    medico VARCHAR(120), lugar_recoleccion VARCHAR(120), lugar_entrega VARCHAR(120),
    categoria VARCHAR(40), subcategoria VARCHAR(60), sitio VARCHAR(80), tipo_muestra VARCHAR(80),
    cantidad INTEGER, citologia_hormonal VARCHAR(3),
    responsable_id INTEGER REFERENCES usuarios(id),
    observaciones TEXT,
    protocolo_sistema VARCHAR(40), cargado_sistema INTEGER NOT NULL DEFAULT 0,
    anulado INTEGER NOT NULL DEFAULT 0,
    creado_por INTEGER REFERENCES usuarios(id), creado_en VARCHAR(19),
    UNIQUE (tipo, protocolo)
);
CREATE TABLE IF NOT EXISTS etapas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    caso_id INTEGER NOT NULL REFERENCES casos(id),
    etapa VARCHAR(30) NOT NULL,
    usuario_id INTEGER NOT NULL REFERENCES usuarios(id),
    fecha_hora VARCHAR(19) NOT NULL,
    UNIQUE (caso_id, etapa)
);
CREATE TABLE IF NOT EXISTS macro (
    caso_id INTEGER PRIMARY KEY REFERENCES casos(id),
    template VARCHAR(120), descripcion TEXT, cassettes INTEGER
);
CREATE TABLE IF NOT EXISTS micro (
    caso_id INTEGER PRIMARY KEY REFERENCES casos(id),
    template VARCHAR(120), descripcion TEXT, conclusion TEXT,
    bethesda VARCHAR(40), tecnicas_especiales VARCHAR(200), solicita_ihq INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS ihq (
    caso_id INTEGER PRIMARY KEY REFERENCES casos(id),
    marcadores TEXT, resultado TEXT
);
CREATE TABLE IF NOT EXISTS templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    clase VARCHAR(10) NOT NULL, titulo VARCHAR(120) NOT NULL, tipo_biopsia VARCHAR(40),
    texto TEXT, conclusion TEXT,
    UNIQUE (clase, titulo)
);
CREATE TABLE IF NOT EXISTS catalogo (
    tipo VARCHAR(3) NOT NULL, categoria VARCHAR(40), subcategoria VARCHAR(60),
    sitio VARCHAR(80), tipo_muestra VARCHAR(80)
);
CREATE TABLE IF NOT EXISTS listas (nombre VARCHAR(30) NOT NULL, orden INTEGER, valor VARCHAR(80));
CREATE TABLE IF NOT EXISTS auditoria (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha_hora VARCHAR(19) NOT NULL, usuario_id INTEGER, caso_id INTEGER,
    accion VARCHAR(40) NOT NULL, detalle TEXT
);
CREATE INDEX IF NOT EXISTS ix_etapas_caso ON etapas(caso_id);
CREATE INDEX IF NOT EXISTS ix_auditoria_caso ON auditoria(caso_id);
"""


def ahora():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _conectar():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=15)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")      # varios usuarios leyendo mientras uno escribe
    return con


def con():
    if "db" not in g:
        g.db = _conectar()
    return g.db


def cerrar(_=None):
    c = g.pop("db", None)
    if c is not None:
        c.close()


def q(sql, params=()):
    return con().execute(sql, params).fetchall()


def uno(sql, params=()):
    return con().execute(sql, params).fetchone()


def ex(sql, params=()):
    c = con()
    cur = c.execute(sql, params)
    c.commit()
    return cur.lastrowid


def auditar(usuario_id, caso_id, accion, detalle=""):
    ex("INSERT INTO auditoria (fecha_hora, usuario_id, caso_id, accion, detalle) VALUES (?,?,?,?,?)",
       (ahora(), usuario_id, caso_id, accion, detalle))


# ---------------------------------------------------------------- inicialización
def inicializar():
    c = _conectar()
    c.executescript(SCHEMA)
    seed = os.path.join(BASE, "seed")

    def cargar(nombre):
        with open(os.path.join(seed, nombre), encoding="utf-8") as f:
            return json.load(f)

    if not c.execute("SELECT 1 FROM catalogo LIMIT 1").fetchone():
        c.executemany("INSERT INTO catalogo VALUES (?,?,?,?,?)",
                      [(x["tipo"], x["categoria"], x["subcategoria"], x["sitio"], x["tipo_muestra"])
                       for x in cargar("catalogo.json")])
    if not c.execute("SELECT 1 FROM listas LIMIT 1").fetchone():
        c.executemany("INSERT INTO listas VALUES (?,?,?)",
                      [(n, i, v) for n, vals in cargar("listas.json").items() for i, v in enumerate(vals)])
    if not c.execute("SELECT 1 FROM templates LIMIT 1").fetchone():
        for clase in ("macro", "micro"):
            c.executemany("INSERT INTO templates (clase, titulo, tipo_biopsia, texto, conclusion) VALUES (?,?,?,?,?)",
                          [(t["clase"], t["titulo"], t["tipo_biopsia"], t["texto"], t["conclusion"])
                           for t in cargar(f"templates_{clase}.json")])
    if not c.execute("SELECT 1 FROM usuarios LIMIT 1").fetchone():
        c.executemany("INSERT INTO usuarios (iniciales, nombre, sectores) VALUES (?,?,?)",
                      [(u["iniciales"], u["iniciales"], ",".join(u["sectores"])) for u in cargar("usuarios.json")])
    c.commit()
    c.close()


# ---------------------------------------------------------------- respaldos
def respaldar(destinos, conservar=48):
    """Copia consistente de la base (API de backup de SQLite) a cada carpeta destino."""
    marca = datetime.now().strftime("%Y%m%d_%H%M")
    origen = sqlite3.connect(DB_PATH)
    try:
        for carpeta in destinos:
            os.makedirs(carpeta, exist_ok=True)
            archivo = os.path.join(carpeta, f"contingencia_{marca}.db")
            dst = sqlite3.connect(archivo)
            origen.backup(dst)
            dst.close()
            viejos = sorted(f for f in os.listdir(carpeta) if f.startswith("contingencia_") and f.endswith(".db"))
            for f in viejos[:-conservar]:
                os.remove(os.path.join(carpeta, f))
    finally:
        origen.close()


def iniciar_respaldos(minutos=10):
    destinos = [os.path.join(DATA, "respaldos")]
    extra = os.environ.get("CONTINGENCIA_RESPALDO")          # ej. carpeta de SharePoint sincronizada
    if extra:
        destinos.append(extra)

    def ciclo():
        while True:
            try:
                respaldar(destinos)
            except Exception as e:                            # un respaldo fallido no debe tirar la app
                print("Respaldo fallido:", e)
            time.sleep(minutos * 60)

    threading.Thread(target=ciclo, daemon=True).start()


def copiar_archivo_db(destino):
    shutil.copy2(DB_PATH, destino)
