"""Sistema web provisorio de contingencia — CAP Vighi.

Corre en una PC del laboratorio; el resto entra por la red interna con el navegador.
Iniciar:  py app.py   (o "Iniciar contingencia.bat")
"""
import io
import os
import secrets
import socket
from datetime import datetime
from functools import wraps

from flask import (Flask, abort, flash, jsonify, redirect, render_template, request, send_file,
                   session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

import db
from flujos import IHQ, NOMBRE_ETAPA, SECTORES, TIPOS, estado, flujo, requisito

PUERTO = int(os.environ.get("CONTINGENCIA_PUERTO", "8000"))
# 1 = solo el sector correspondiente puede marcar cada etapa como lista (el admin siempre puede)
ESTRICTO = os.environ.get("CONTINGENCIA_ESTRICTO", "0") == "1"

app = Flask(__name__)


def _clave_secreta():
    ruta = os.path.join(db.DATA, "secret.key")
    os.makedirs(db.DATA, exist_ok=True)
    if not os.path.exists(ruta):
        with open(ruta, "w") as f:
            f.write(secrets.token_hex(32))
    with open(ruta) as f:
        return f.read().strip()


app.secret_key = _clave_secreta()
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")
app.teardown_appcontext(db.cerrar)


# ---------------------------------------------------------------- sesión y seguridad
def usuario_actual():
    uid = session.get("uid")
    return db.uno("SELECT * FROM usuarios WHERE id=? AND activo=1", (uid,)) if uid else None


def requiere_login(f):
    @wraps(f)
    def envuelta(*a, **k):
        u = usuario_actual()
        if not u:
            return redirect(url_for("login", next=request.path))
        if u["debe_cambiar_clave"] and request.endpoint not in ("cambiar_clave", "logout"):
            return redirect(url_for("cambiar_clave"))
        return f(*a, **k)
    return envuelta


def requiere_admin(f):
    @wraps(f)
    @requiere_login
    def envuelta(*a, **k):
        if not usuario_actual()["admin"]:
            abort(403)
        return f(*a, **k)
    return envuelta


@app.before_request
def verificar_csrf():
    if "csrf" not in session:
        session["csrf"] = secrets.token_hex(16)
    if request.method == "POST":
        token = request.form.get("_csrf") or request.headers.get("X-CSRF")
        if token != session["csrf"]:
            abort(400, "Formulario vencido: recargá la página e intentá de nuevo.")


@app.context_processor
def globales():
    u = usuario_actual()
    return {"yo": u, "csrf": session.get("csrf", ""), "TIPOS": TIPOS, "SECTORES": SECTORES,
            "NOMBRE_ETAPA": NOMBRE_ETAPA, "mis_sectores": set((u["sectores"] or "").split(",")) if u else set()}


@app.template_filter("fh")
def fecha_hora(v):
    if not v:
        return ""
    try:
        return datetime.strptime(v, "%Y-%m-%d %H:%M:%S").strftime("%d/%m/%Y %H:%M")
    except ValueError:
        return v


@app.template_filter("f")
def fecha(v):
    if not v:
        return ""
    try:
        return datetime.strptime(v, "%Y-%m-%d").strftime("%d/%m/%Y")
    except ValueError:
        return v


def es_local():
    return request.remote_addr in ("127.0.0.1", "::1")


# ---------------------------------------------------------------- login
@app.route("/configurar", methods=["GET", "POST"])
def configurar():
    """Primer arranque: crear la clave del administrador. Solo desde la PC servidor."""
    if db.uno("SELECT 1 FROM usuarios WHERE admin=1 AND clave_hash IS NOT NULL"):
        return redirect(url_for("login"))
    if not es_local():
        abort(403, "La configuración inicial solo se hace desde la PC servidor.")
    if request.method == "POST":
        ini = request.form["iniciales"].strip().upper()
        clave = request.form["clave"]
        if len(clave) < 8 or clave != request.form["clave2"]:
            flash("La clave debe tener al menos 8 caracteres y coincidir en los dos campos.", "error")
        elif not ini:
            flash("Faltan las iniciales.", "error")
        else:
            existente = db.uno("SELECT id FROM usuarios WHERE iniciales=?", (ini,))
            h = generate_password_hash(clave)
            if existente:
                db.ex("UPDATE usuarios SET admin=1, activo=1, clave_hash=?, debe_cambiar_clave=0, nombre=? WHERE id=?",
                      (h, request.form["nombre"].strip() or ini, existente["id"]))
            else:
                db.ex("INSERT INTO usuarios (iniciales, nombre, sectores, admin, clave_hash, debe_cambiar_clave) "
                      "VALUES (?,?,?,1,?,0)", (ini, request.form["nombre"].strip() or ini, ",".join(SECTORES), h))
            flash("Administrador creado. Ya podés ingresar.", "ok")
            return redirect(url_for("login"))
    return render_template("configurar.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if not db.uno("SELECT 1 FROM usuarios WHERE admin=1 AND clave_hash IS NOT NULL"):
        return redirect(url_for("configurar"))
    if request.method == "POST":
        u = db.uno("SELECT * FROM usuarios WHERE iniciales=? AND activo=1", (request.form["iniciales"].strip().upper(),))
        if u and u["clave_hash"] and check_password_hash(u["clave_hash"], request.form["clave"]):
            session.clear()
            session["uid"] = u["id"]
            db.auditar(u["id"], None, "login")
            destino = request.args.get("next", "")
            return redirect(destino if destino.startswith("/") and not destino.startswith("//") else url_for("tablero"))
        flash("Iniciales o clave incorrectas (o el usuario todavía no tiene clave: pedísela al administrador).", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/clave", methods=["GET", "POST"])
@requiere_login
def cambiar_clave():
    yo = usuario_actual()
    if request.method == "POST":
        nueva = request.form["nueva"]
        if not check_password_hash(yo["clave_hash"], request.form["actual"]):
            flash("La clave actual no es correcta.", "error")
        elif len(nueva) < 8 or nueva != request.form["nueva2"]:
            flash("La nueva clave debe tener al menos 8 caracteres y coincidir en los dos campos.", "error")
        else:
            db.ex("UPDATE usuarios SET clave_hash=?, debe_cambiar_clave=0 WHERE id=?", (generate_password_hash(nueva), yo["id"]))
            db.auditar(yo["id"], None, "cambio_clave")
            flash("Clave actualizada.", "ok")
            return redirect(url_for("tablero"))
    return render_template("clave.html")


# ---------------------------------------------------------------- casos
def cargar_casos(where="1=1", params=()):
    casos = db.q(f"""SELECT c.*, u.iniciales AS responsable, COALESCE(m.solicita_ihq,0) AS solicita_ihq
                     FROM casos c LEFT JOIN usuarios u ON u.id=c.responsable_id
                     LEFT JOIN micro m ON m.caso_id=c.id WHERE {where} ORDER BY c.id DESC""", params)
    hechas = {}
    for e in db.q("SELECT caso_id, etapa FROM etapas"):
        hechas.setdefault(e["caso_id"], set()).add(e["etapa"])
    out = []
    for c in casos:
        d = dict(c)
        d["estado"], prox = estado(c["tipo"], hechas.get(c["id"], set()), bool(c["solicita_ihq"]), bool(c["anulado"]))
        d["proxima"] = prox
        out.append(d)
    return out


@app.route("/")
@requiere_login
def tablero():
    yo = usuario_actual()
    f = {k: request.args.get(k, "") for k in ("q", "tipo", "ver", "sector")}
    f["ver"] = f["ver"] or "pendientes"
    casos = cargar_casos()
    conteo = {"pendientes": 0, "informados": 0, "sin_cargar": 0}
    for c in casos:
        if c["anulado"]:
            continue
        if c["proxima"]:
            conteo["pendientes"] += 1
        else:
            conteo["informados"] += 1
            if not c["cargado_sistema"]:
                conteo["sin_cargar"] += 1
    mis = set((yo["sectores"] or "").split(","))

    def pasa(c):
        if f["tipo"] and c["tipo"] != f["tipo"]:
            return False
        if f["ver"] == "pendientes" and (c["anulado"] or not c["proxima"]):
            return False
        if f["ver"] == "mios" and (c["anulado"] or not c["proxima"] or c["proxima"][2] not in mis):
            return False
        if f["ver"] == "informados" and (c["anulado"] or c["proxima"]):
            return False
        if f["ver"] == "sin_cargar" and (c["anulado"] or c["proxima"] or c["cargado_sistema"]):
            return False
        if f["ver"] == "anulados" and not c["anulado"]:
            return False
        if f["sector"] and not (c["proxima"] and c["proxima"][2] == f["sector"]):
            return False
        if f["q"]:
            texto = " ".join(str(c[k] or "") for k in ("protocolo", "apellido", "nombre", "dni", "lote", "sitio")).lower()
            return f["q"].lower() in texto
        return True

    return render_template("tablero.html", casos=[c for c in casos if pasa(c)], f=f, conteo=conteo)


CAMPOS = ["fecha_recoleccion", "dni", "cobertura", "n_afiliado", "nombre", "apellido", "sexo", "exento",
          "fecha_nacimiento", "email", "telefono", "medico", "lugar_recoleccion", "lugar_entrega",
          "subcategoria", "sitio", "tipo_muestra", "cantidad", "citologia_hormonal", "observaciones"]


def listas():
    out = {}
    for r in db.q("SELECT nombre, valor FROM listas ORDER BY nombre, orden"):
        out.setdefault(r["nombre"], []).append(r["valor"])
    return out


def responsables(tipo):
    sector = "firmante" if tipo == "BP" else "citotecnico"
    return [u for u in db.q("SELECT id, iniciales, nombre, sectores FROM usuarios WHERE activo=1 ORDER BY iniciales")
            if sector in (u["sectores"] or "").split(",")]


def datos_formulario(tipo):
    d = {k: request.form.get(k, "").strip() for k in CAMPOS}
    d["protocolo"] = request.form.get("protocolo", "").strip()
    d["cantidad"] = int(d["cantidad"]) if d["cantidad"].isdigit() else None
    d["responsable_id"] = int(request.form["responsable_id"]) if request.form.get("responsable_id", "").isdigit() else None
    d["categoria"] = "Biopsias" if tipo == "BP" else "Citologías"
    if tipo == "PAP":
        d["subcategoria"], d["sitio"] = "Ginecológica", "Vagina"
    if tipo == "BP":
        d["tipo_lote"] = request.form.get("tipo_lote", "").strip() or None
        d["n_lote"] = int(request.form["n_lote"]) if request.form.get("n_lote", "").isdigit() else 1
    d["lote_manual"] = request.form.get("lote", "").strip()
    return d


@app.route("/ingreso/<tipo>", methods=["GET", "POST"])
@app.route("/caso/<int:caso_id>/editar", methods=["GET", "POST"])
@requiere_login
def ingreso(tipo=None, caso_id=None):
    yo = usuario_actual()
    caso = db.uno("SELECT * FROM casos WHERE id=?", (caso_id,)) if caso_id else None
    if caso_id and not caso:
        abort(404)
    tipo = caso["tipo"] if caso else tipo
    if tipo not in TIPOS:
        abort(404)
    if request.method == "POST":
        d = datos_formulario(tipo)
        dup = db.uno("SELECT id FROM casos WHERE tipo=? AND protocolo=? AND id<>?", (tipo, d["protocolo"], caso_id or 0))
        if not d["protocolo"]:
            flash("Falta el N° de protocolo.", "error")
        elif dup:
            flash(f"El protocolo {d['protocolo']} ya existe en {TIPOS[tipo]}.", "error")
        else:
            if tipo == "BP":
                if d["lote_manual"] and caso:                       # al editar se puede corregir el lote a mano
                    d["lote"] = d["lote_manual"]
                elif d["tipo_lote"]:
                    base = caso["creado_en"] if caso else db.ahora()
                    d["lote"] = f"{d['tipo_lote']}-{base[5:7]}{base[8:10]}.{d['n_lote']}"
                else:
                    d["lote"] = d["lote_manual"] or None
            else:
                d["lote"] = d["lote_manual"] or None
            cols = ["protocolo", "lote", "categoria", "responsable_id"] + CAMPOS + (["tipo_lote", "n_lote"] if tipo == "BP" else [])
            if caso:
                db.ex(f"UPDATE casos SET {', '.join(c + '=?' for c in cols)} WHERE id=?", [d[c] for c in cols] + [caso_id])
                db.auditar(yo["id"], caso_id, "editar_ingreso")
                flash("Datos actualizados.", "ok")
            else:
                caso_id = db.ex(f"INSERT INTO casos (tipo, creado_por, creado_en, {', '.join(cols)}) "
                                f"VALUES (?,?,?,{','.join('?' * len(cols))})", [tipo, yo["id"], db.ahora()] + [d[c] for c in cols])
                db.auditar(yo["id"], caso_id, "ingreso", d["protocolo"])
                flash(f"Protocolo {d['protocolo']} ingresado.", "ok")
            return redirect(url_for("caso", caso_id=caso_id))
        caso = {**(dict(caso) if caso else {}), **d}
    ultimos = db.q("SELECT protocolo FROM casos WHERE tipo=? ORDER BY id DESC LIMIT 5", (tipo,))
    return render_template("ingreso.html", tipo=tipo, caso=caso, listas=listas(), responsables=responsables(tipo),
                           ultimos=[r["protocolo"] for r in ultimos], editando=bool(caso_id))


def ficha(caso_id):
    c = db.uno("""SELECT c.*, u.iniciales AS responsable, u.nombre AS responsable_nombre
                  FROM casos c LEFT JOIN usuarios u ON u.id=c.responsable_id WHERE c.id=?""", (caso_id,))
    if not c:
        abort(404)
    macro = db.uno("SELECT * FROM macro WHERE caso_id=?", (caso_id,))
    micro = db.uno("SELECT * FROM micro WHERE caso_id=?", (caso_id,))
    ihq = db.uno("SELECT * FROM ihq WHERE caso_id=?", (caso_id,))
    etapas = {e["etapa"]: e for e in db.q("""SELECT e.*, u.iniciales, u.nombre FROM etapas e
                                            JOIN usuarios u ON u.id=e.usuario_id WHERE caso_id=?""", (caso_id,))}
    solicita = bool(micro and micro["solicita_ihq"])
    return c, macro, micro, ihq, etapas, solicita


@app.route("/caso/<int:caso_id>")
@requiere_login
def caso(caso_id):
    c, macro, micro, ihq, etapas, solicita = ficha(caso_id)
    est, prox = estado(c["tipo"], set(etapas), solicita, bool(c["anulado"]))
    ultima = db.uno("SELECT * FROM etapas WHERE caso_id=? ORDER BY fecha_hora DESC, id DESC LIMIT 1", (caso_id,))
    historial = db.q("""SELECT a.*, u.iniciales FROM auditoria a LEFT JOIN usuarios u ON u.id=a.usuario_id
                        WHERE caso_id=? ORDER BY a.id DESC""", (caso_id,))
    tpl = lambda clase: db.q("SELECT id, titulo FROM templates WHERE clase=? ORDER BY titulo", (clase,))
    return render_template("caso.html", c=c, macro=macro, micro=micro, ihq=ihq, etapas=etapas, estado=est, prox=prox,
                           pasos=flujo(c["tipo"], solicita), ultima=ultima, historial=historial,
                           tpl_macro=tpl("macro"), tpl_micro=tpl("micro"), bethesda=listas().get("bethesda", []),
                           ihq_hecha=any(k in etapas for k, _, _ in IHQ))


def guardar_seccion(tabla, caso_id, valores):
    if db.uno(f"SELECT 1 FROM {tabla} WHERE caso_id=?", (caso_id,)):
        db.ex(f"UPDATE {tabla} SET {', '.join(k + '=?' for k in valores)} WHERE caso_id=?", list(valores.values()) + [caso_id])
    else:
        db.ex(f"INSERT INTO {tabla} (caso_id, {', '.join(valores)}) VALUES (?,{','.join('?' * len(valores))})",
              [caso_id] + list(valores.values()))


@app.route("/caso/<int:caso_id>/macro", methods=["POST"])
@requiere_login
def guardar_macro(caso_id):
    ficha(caso_id)
    cas = request.form.get("cassettes", "")
    guardar_seccion("macro", caso_id, {"template": request.form.get("template", "").strip(),
                                       "descripcion": request.form.get("descripcion", "").strip(),
                                       "cassettes": int(cas) if cas.isdigit() else None})
    db.auditar(usuario_actual()["id"], caso_id, "guardar_macro")
    flash("Macroscopía guardada.", "ok")
    if request.form.get("y_listo"):
        return marcar_listo(caso_id, "macroscopia")
    return redirect(url_for("caso", caso_id=caso_id) + "#macro")


@app.route("/caso/<int:caso_id>/micro", methods=["POST"])
@requiere_login
def guardar_micro(caso_id):
    c, _, micro, _, etapas, solicita = ficha(caso_id)
    pide_ihq = 1 if request.form.get("solicita_ihq") == "1" else 0
    if solicita and not pide_ihq and any(k in etapas for k, _, _ in IHQ):
        flash("No se puede quitar la IHQ: ya tiene etapas de IHQ registradas.", "error")
        pide_ihq = 1
    guardar_seccion("micro", caso_id, {"template": request.form.get("template", "").strip(),
                                       "descripcion": request.form.get("descripcion", "").strip(),
                                       "conclusion": request.form.get("conclusion", "").strip(),
                                       "bethesda": request.form.get("bethesda", "").strip(),
                                       "tecnicas_especiales": request.form.get("tecnicas_especiales", "").strip(),
                                       "solicita_ihq": pide_ihq if c["tipo"] == "BP" else 0})
    db.auditar(usuario_actual()["id"], caso_id, "guardar_micro", "solicita IHQ" if pide_ihq else "")
    flash("Microscopía guardada.", "ok")
    if request.form.get("y_listo"):
        return marcar_listo(caso_id, "microscopia")
    return redirect(url_for("caso", caso_id=caso_id) + "#micro")


@app.route("/caso/<int:caso_id>/ihq", methods=["POST"])
@requiere_login
def guardar_ihq(caso_id):
    ficha(caso_id)
    guardar_seccion("ihq", caso_id, {"marcadores": request.form.get("marcadores", "").strip(),
                                     "resultado": request.form.get("resultado", "").strip()})
    db.auditar(usuario_actual()["id"], caso_id, "guardar_ihq")
    flash("IHQ guardada.", "ok")
    if request.form.get("y_listo"):
        return marcar_listo(caso_id, "interp_ihq")
    return redirect(url_for("caso", caso_id=caso_id) + "#ihq")


@app.route("/caso/<int:caso_id>/listo/<etapa>", methods=["POST"])
@requiere_login
def marcar_listo(caso_id, etapa):
    yo = usuario_actual()
    c, macro, micro, ihq, etapas, solicita = ficha(caso_id)
    _, prox = estado(c["tipo"], set(etapas), solicita, bool(c["anulado"]))
    if not prox or prox[0] != etapa:
        flash("Esa etapa no es la próxima pendiente de este caso.", "error")
    elif ESTRICTO and not yo["admin"] and prox[2] not in (yo["sectores"] or "").split(","):
        flash(f"La etapa {prox[1]} la marca el sector {SECTORES[prox[2]]}.", "error")
    elif (falta := requisito(c["tipo"], etapa, macro, micro, ihq)):
        flash(falta, "error")
    else:
        db.ex("INSERT INTO etapas (caso_id, etapa, usuario_id, fecha_hora) VALUES (?,?,?,?)", (caso_id, etapa, yo["id"], db.ahora()))
        db.auditar(yo["id"], caso_id, "listo", NOMBRE_ETAPA[etapa])
        flash(f"✔ {NOMBRE_ETAPA[etapa]} registrada por {yo['iniciales']}.", "ok")
    return redirect(request.form.get("volver") or url_for("caso", caso_id=caso_id))


@app.route("/caso/<int:caso_id>/deshacer", methods=["POST"])
@requiere_login
def deshacer(caso_id):
    yo = usuario_actual()
    ultima = db.uno("SELECT * FROM etapas WHERE caso_id=? ORDER BY fecha_hora DESC, id DESC LIMIT 1", (caso_id,))
    if not ultima:
        flash("No hay etapas para deshacer.", "error")
    elif ultima["usuario_id"] != yo["id"] and not yo["admin"]:
        flash("Solo quien la registró o un administrador puede deshacer la etapa.", "error")
    else:
        db.ex("DELETE FROM etapas WHERE id=?", (ultima["id"],))
        db.auditar(yo["id"], caso_id, "deshacer", NOMBRE_ETAPA.get(ultima["etapa"], ultima["etapa"]))
        flash(f"Se deshizo {NOMBRE_ETAPA.get(ultima['etapa'])}.", "ok")
    return redirect(url_for("caso", caso_id=caso_id))


@app.route("/caso/<int:caso_id>/sistema", methods=["POST"])
@requiere_login
def cargado_sistema(caso_id):
    ficha(caso_id)
    cargado = 1 if request.form.get("cargado_sistema") == "1" else 0
    db.ex("UPDATE casos SET protocolo_sistema=?, cargado_sistema=? WHERE id=?",
          (request.form.get("protocolo_sistema", "").strip(), cargado, caso_id))
    db.auditar(usuario_actual()["id"], caso_id, "sistema", "cargado" if cargado else "no cargado")
    flash("Datos de carga en el sistema actualizados.", "ok")
    return redirect(url_for("caso", caso_id=caso_id))


@app.route("/caso/<int:caso_id>/anular", methods=["POST"])
@requiere_login
def anular(caso_id):
    c = ficha(caso_id)[0]
    motivo = request.form.get("motivo", "").strip()
    nuevo = 0 if c["anulado"] else 1
    if nuevo and not motivo:
        flash("Indicá el motivo de la anulación.", "error")
    else:
        db.ex("UPDATE casos SET anulado=? WHERE id=?", (nuevo, caso_id))
        db.auditar(usuario_actual()["id"], caso_id, "anular" if nuevo else "reactivar", motivo)
        flash("Caso anulado." if nuevo else "Caso reactivado.", "ok")
    return redirect(url_for("caso", caso_id=caso_id))


# ---------------------------------------------------------------- APIs para los formularios
@app.route("/api/catalogo/<tipo>")
@requiere_login
def api_catalogo(tipo):
    arbol = {}
    for r in db.q("SELECT subcategoria, sitio, tipo_muestra FROM catalogo WHERE tipo=?", (tipo,)):
        arbol.setdefault(r["subcategoria"], {}).setdefault(r["sitio"], []).append(r["tipo_muestra"])
    return jsonify(arbol)


@app.route("/api/template/<int:tid>")
@requiere_login
def api_template(tid):
    t = db.uno("SELECT titulo, texto, conclusion FROM templates WHERE id=?", (tid,))
    return jsonify(dict(t)) if t else (jsonify({}), 404)


# ---------------------------------------------------------------- usuarios
@app.route("/usuarios")
@requiere_admin
def usuarios():
    return render_template("usuarios.html", usuarios=db.q("SELECT * FROM usuarios ORDER BY activo DESC, iniciales"))


@app.route("/usuarios/nuevo", methods=["GET", "POST"])
@app.route("/usuarios/<int:uid>", methods=["GET", "POST"])
@requiere_admin
def usuario(uid=None):
    yo = usuario_actual()
    u = db.uno("SELECT * FROM usuarios WHERE id=?", (uid,)) if uid else None
    if request.method == "POST":
        ini = request.form["iniciales"].strip().upper()
        sectores = ",".join(s for s in SECTORES if request.form.get("s_" + s))
        vals = (ini, request.form.get("nombre", "").strip() or ini, sectores,
                1 if request.form.get("admin") else 0, 1 if request.form.get("activo") else 0)
        dup = db.uno("SELECT id FROM usuarios WHERE iniciales=? AND id<>?", (ini, uid or 0))
        clave = request.form.get("clave", "")
        if not ini or dup:
            flash("Iniciales vacías o ya usadas por otra persona.", "error")
        elif clave and len(clave) < 8:
            flash("La clave temporal debe tener al menos 8 caracteres.", "error")
        elif uid == yo["id"] and not (vals[3] and vals[4]):
            flash("No podés quitarte el rol de administrador ni desactivarte a vos mismo.", "error")
        else:
            if u:
                db.ex("UPDATE usuarios SET iniciales=?, nombre=?, sectores=?, admin=?, activo=? WHERE id=?", vals + (uid,))
            else:
                uid = db.ex("INSERT INTO usuarios (iniciales, nombre, sectores, admin, activo) VALUES (?,?,?,?,?)", vals)
            if clave:
                db.ex("UPDATE usuarios SET clave_hash=?, debe_cambiar_clave=1 WHERE id=?", (generate_password_hash(clave), uid))
            db.auditar(yo["id"], None, "usuario", f"{ini}{' (clave temporal)' if clave else ''}")
            flash(f"Usuario {ini} guardado." + (" Al ingresar va a tener que cambiar la clave." if clave else ""), "ok")
            return redirect(url_for("usuarios"))
    return render_template("usuario.html", u=u)


# ---------------------------------------------------------------- exportar
@app.route("/exportar")
@requiere_login
def exportar():
    """Excel con todos los casos y sus etapas, para volver a cargarlos en el sistema."""
    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    etapas = {}
    for e in db.q("SELECT e.caso_id, e.etapa, e.fecha_hora, u.iniciales FROM etapas e JOIN usuarios u ON u.id=e.usuario_id"):
        etapas.setdefault(e["caso_id"], {})[e["etapa"]] = e
    macro = {r["caso_id"]: r for r in db.q("SELECT * FROM macro")}
    micro = {r["caso_id"]: r for r in db.q("SELECT * FROM micro")}
    ihq = {r["caso_id"]: r for r in db.q("SELECT * FROM ihq")}
    orden_etapas = list(dict.fromkeys(k for k in NOMBRE_ETAPA))

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Casos"
    cab = (["Tipo", "N° Protocolo", "Estado", "Lote", "Anulado", "Protocolo sistema", "Cargado en sistema", "Responsable",
            "Ingresado", "Fecha recolección", "DNI", "Apellido", "Nombre", "Sexo", "Fecha nacimiento", "Cobertura",
            "N° afiliado", "Exento", "E-mail", "Teléfono", "Médico", "Lugar recolección", "Lugar entrega", "Categoría",
            "Subcategoría", "Sitio", "Tipo de muestra", "Cantidad", "Citología hormonal", "Observaciones",
            "Template macro", "Descripción macro", "Cassettes", "Template micro", "Descripción micro", "Conclusión",
            "Bethesda", "Técnicas especiales", "Solicita IHQ", "Marcadores IHQ", "Resultado IHQ"]
           + [f"{NOMBRE_ETAPA[k]} - {x}" for k in orden_etapas for x in ("usuario", "fecha/hora")])
    ws.append(cab)
    for c in cargar_casos():
        ma, mi, ih, et = macro.get(c["id"]), micro.get(c["id"]), ihq.get(c["id"]), etapas.get(c["id"], {})
        g = lambda r, k: (r[k] if r else "") or ""
        fila = [c["tipo"], c["protocolo"], c["estado"], c["lote"], "Si" if c["anulado"] else "", c["protocolo_sistema"],
                "Si" if c["cargado_sistema"] else "No", c["responsable"], fecha_hora(c["creado_en"]),
                fecha(c["fecha_recoleccion"]), c["dni"], c["apellido"], c["nombre"], c["sexo"], fecha(c["fecha_nacimiento"]),
                c["cobertura"], c["n_afiliado"], c["exento"], c["email"], c["telefono"], c["medico"], c["lugar_recoleccion"],
                c["lugar_entrega"], c["categoria"], c["subcategoria"], c["sitio"], c["tipo_muestra"], c["cantidad"],
                c["citologia_hormonal"], c["observaciones"], g(ma, "template"), g(ma, "descripcion"), g(ma, "cassettes"),
                g(mi, "template"), g(mi, "descripcion"), g(mi, "conclusion"), g(mi, "bethesda"),
                g(mi, "tecnicas_especiales"), "Si" if mi and mi["solicita_ihq"] else "", g(ih, "marcadores"), g(ih, "resultado")]
        for k in orden_etapas:
            e = et.get(k)
            fila += [e["iniciales"] if e else "", fecha_hora(e["fecha_hora"]) if e else ""]
        ws.append(fila)
    for i, cell in enumerate(ws[1], start=1):
        cell.font, cell.fill = Font(bold=True, color="FFFFFF"), PatternFill("solid", fgColor="5B2C83")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[get_column_letter(i)].width = 16
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = ws.dimensions
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    db.auditar(usuario_actual()["id"], None, "exportar")
    return send_file(buf, as_attachment=True, download_name=f"contingencia_{datetime.now():%Y%m%d_%H%M}.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ---------------------------------------------------------------- arranque
def ips_locales():
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


if __name__ == "__main__":
    from waitress import serve
    db.inicializar()
    db.iniciar_respaldos(minutos=10)
    print("=" * 64)
    print(" Sistema de contingencia CAP Vighi")
    print(f" En esta PC:        http://localhost:{PUERTO}")
    for ip in ips_locales():
        print(f" Desde otras PCs:   http://{ip}:{PUERTO}")
    print(f" Base de datos:     {db.DB_PATH}")
    print(" Para detenerlo: cerrar esta ventana.")
    print("=" * 64)
    serve(app, host="0.0.0.0", port=PUERTO, threads=12)
