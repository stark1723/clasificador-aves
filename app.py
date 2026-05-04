import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import gdown
import hashlib
import json
import re
import secrets
import string
import sqlite3
from datetime import datetime, timedelta
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AvisFauna — Identificador de Aves",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# CARGA DE ESTILOS DESDE ARCHIVO EXTERNO
# ═══════════════════════════════════════════════════════════════════════════════
def load_css():
    """Carga el archivo static/styles.css e inyecta en la app."""
    css_path = Path(__file__).parent / "static" / "styles.css"
    if css_path.exists():
        with open(css_path) as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ No se encontró static/styles.css")

load_css()

# ═══════════════════════════════════════════════════════════════════════════════
# ROLES Y PERMISOS
# ═══════════════════════════════════════════════════════════════════════════════
ADMIN_EMAIL = "adminfs01@gmail.com"
ROLES = {
    "administrador": {"label":"Administrador","css":"role-admin","icon":"👑"},
    "experto":       {"label":"Experto","css":"role-experto","icon":"🔬"},
    "usuario":       {"label":"Usuario","css":"role-usuario","icon":"🙋"},
}
ROLE_PERMISSIONS = {
    "administrador": ["clasificar","ver_estadisticas","dashboard_admin","gestionar_usuarios"],
    "experto":       ["clasificar","ver_estadisticas"],
    "usuario":       ["clasificar"],
}
def can(role, permission):
    return permission in ROLE_PERMISSIONS.get(role, [])
def role_badge_html(role):
    r = ROLES.get(role, ROLES["usuario"])
    return f'<span class="{r["css"]}">{r["icon"]} {r["label"]}</span>'
def avatar_color(role):
    return {"administrador":"#b91c1c","experto":"#15803d","usuario":"#1d4ed8"}.get(role,"#1d4ed8")

# ═══════════════════════════════════════════════════════════════════════════════
# BASE DE DATOS SQLITE
# ═══════════════════════════════════════════════════════════════════════════════
DB_FILE = "avisfauna.db"

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            email      TEXT PRIMARY KEY,
            name       TEXT NOT NULL,
            password   TEXT NOT NULL,
            role       TEXT NOT NULL DEFAULT 'usuario',
            created_at TEXT NOT NULL,
            last_login TEXT
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            species    TEXT NOT NULL,
            confidence REAL NOT NULL,
            model      TEXT NOT NULL,
            timestamp  TEXT NOT NULL,
            FOREIGN KEY (user_email) REFERENCES users(email)
        );
        CREATE TABLE IF NOT EXISTS reset_tokens (
            email   TEXT PRIMARY KEY,
            token   TEXT NOT NULL,
            expires TEXT NOT NULL
        );
        """)

# ── Usuarios ──────────────────────────────────────────────────────────────────
def db_get_user(email):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        return dict(row) if row else None

def db_get_all_users():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

def db_create_user(email, name, password_hash, role, created_at):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO users (email,name,password,role,created_at) VALUES (?,?,?,?,?)",
            (email, name, password_hash, role, created_at)
        )

def db_update_last_login(email):
    with get_db() as conn:
        conn.execute("UPDATE users SET last_login=? WHERE email=?", (datetime.now().isoformat(), email))

def db_update_role(email, new_role):
    with get_db() as conn:
        conn.execute("UPDATE users SET role=? WHERE email=?", (new_role, email))

def db_update_password(email, new_hash):
    with get_db() as conn:
        conn.execute("UPDATE users SET password=? WHERE email=?", (new_hash, email))

def db_ensure_admin():
    with get_db() as conn:
        conn.execute("UPDATE users SET role='administrador' WHERE email=?", (ADMIN_EMAIL,))

# ── Predicciones ──────────────────────────────────────────────────────────────
def db_record_prediction(user_email, species, confidence, model_used):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO predictions (user_email,species,confidence,model,timestamp) VALUES (?,?,?,?,?)",
            (user_email, species, round(confidence,2), model_used, datetime.now().isoformat())
        )

def db_get_predictions(user_email=None, limit=None):
    with get_db() as conn:
        if user_email:
            q, args = "SELECT * FROM predictions WHERE user_email=? ORDER BY timestamp DESC", (user_email,)
        else:
            q, args = "SELECT * FROM predictions ORDER BY timestamp DESC", ()
        if limit:
            q += f" LIMIT {limit}"
        return [dict(r) for r in conn.execute(q, args).fetchall()]

def db_count_predictions(user_email=None):
    with get_db() as conn:
        if user_email:
            return conn.execute("SELECT COUNT(*) FROM predictions WHERE user_email=?", (user_email,)).fetchone()[0]
        return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

def db_top_species(user_email=None, limit=5):
    with get_db() as conn:
        if user_email:
            rows = conn.execute(
                "SELECT species,COUNT(*) as cnt FROM predictions WHERE user_email=? GROUP BY species ORDER BY cnt DESC LIMIT ?",
                (user_email, limit)).fetchall()
        else:
            rows = conn.execute(
                "SELECT species,COUNT(*) as cnt FROM predictions GROUP BY species ORDER BY cnt DESC LIMIT ?",
                (limit,)).fetchall()
        return [(r["species"], r["cnt"]) for r in rows]

def db_avg_confidence(user_email=None):
    with get_db() as conn:
        if user_email:
            r = conn.execute("SELECT AVG(confidence) FROM predictions WHERE user_email=?", (user_email,)).fetchone()[0]
        else:
            r = conn.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0]
        return round(r, 1) if r else 0

# ── Reset tokens ──────────────────────────────────────────────────────────────
def db_set_reset_token(email, token, expires):
    with get_db() as conn:
        conn.execute("INSERT OR REPLACE INTO reset_tokens (email,token,expires) VALUES (?,?,?)", (email, token, expires))

def db_get_reset_token(email):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM reset_tokens WHERE email=?", (email,)).fetchone()
        return dict(row) if row else None

def db_delete_reset_token(email):
    with get_db() as conn:
        conn.execute("DELETE FROM reset_tokens WHERE email=?", (email,))

# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════════
def hash_password(p): return hashlib.sha256(p.encode()).hexdigest()
def validate_email(e): return re.match(r'^[\w\.-]+@[\w\.-]+\.\w{2,}$', e)
def generate_token(n=8): return ''.join(secrets.choice(string.ascii_letters+string.digits) for _ in range(n))
def get_user_role(email):
    if email == ADMIN_EMAIL: return "administrador"
    u = db_get_user(email)
    return u.get("role","usuario") if u else "usuario"

# ═══════════════════════════════════════════════════════════════════════════════
# ENVÍO DE CORREO SMTP
# ═══════════════════════════════════════════════════════════════════════════════
def get_email_config():
    try: return st.secrets["email"]["sender"], st.secrets["email"]["password"]
    except Exception: return None, None

def send_reset_email(dest_email, dest_name, token):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    sender, password = get_email_config()
    if not sender or not password: return False, "no_config"
    html = f"""<!DOCTYPE html><html><body style="background:#0a1628;font-family:Arial,sans-serif;margin:0;padding:40px;">
    <div style="max-width:520px;margin:0 auto;background:#0f1f38;border-radius:16px;overflow:hidden;border:1px solid rgba(255,255,255,0.1);">
    <div style="background:linear-gradient(135deg,#1d4ed8,#2563eb);padding:32px;text-align:center;">
    <div style="font-size:2rem;">🦅</div><div style="font-family:Georgia,serif;font-size:1.5rem;font-weight:700;color:#fff;margin-top:8px;">AvisFauna</div>
    </div><div style="padding:36px;">
    <p style="color:rgba(255,255,255,0.85);font-size:0.97rem;line-height:1.6;">Hola <strong style="color:#fff;">{dest_name}</strong>,</p>
    <p style="color:rgba(255,255,255,0.65);font-size:0.92rem;line-height:1.6;">Tu codigo de recuperacion es:</p>
    <div style="background:rgba(37,99,235,0.15);border:2px solid rgba(96,165,250,0.35);border-radius:12px;padding:24px;text-align:center;margin:20px 0;">
    <div style="font-size:2.4rem;font-weight:700;color:#60a5fa;letter-spacing:0.3em;font-family:'Courier New',monospace;">{token}</div>
    <div style="font-size:0.78rem;color:rgba(255,255,255,0.4);margin-top:10px;">Valido durante 15 minutos</div>
    </div></div></div></body></html>"""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "AvisFauna - Codigo de recuperacion"
    msg["From"] = f"AvisFauna <{sender}>"
    msg["To"] = dest_email
    msg.attach(MIMEText(html, "html", "utf-8"))
    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as srv:
            srv.ehlo(); srv.starttls(); srv.login(sender, password)
            srv.sendmail(sender, dest_email, msg.as_string())
        return True, ""
    except smtplib.SMTPAuthenticationError: return False, "auth_error"
    except Exception as e: return False, str(e)

# ═══════════════════════════════════════════════════════════════════════════════
# MODELOS GOOGLE DRIVE
# ═══════════════════════════════════════════════════════════════════════════════
GDRIVE_MODEL_1_ID = '1FcU2jJEYqU0c971n9WIytWfFaTD018z-'
GDRIVE_MODEL_2_ID = '1ZvA60kdraCRDw8wGx83FqfUkwWOeykPC'
MODEL_1_PATH = 'inception_finetuned.h5'
MODEL_2_PATH = 'VGG16_finetuned.h5'
IMG_SIZE = (224, 224)
CLASSES_MODEL_1 = ['Anhinga anhinga','Butorides striata','Chamaepetes goudotii','Colinus cristatus','Nycticorax nycticorax','Ortalis guttata','Penelope montagnii','Podilymbus podiceps','Tachybaptus dominicus','Tigrisoma lineatum']
CLASSES_MODEL_2 = CLASSES_MODEL_1.copy()
COMMON_NAMES = {'Anhinga anhinga':'Pato aguja','Butorides striata':'Garza verdosa','Chamaepetes goudotii':'Pava caucana','Colinus cristatus':'Codorniz crestada','Nycticorax nycticorax':'Garza nocturna','Ortalis guttata':'Guacharaca moteada','Penelope montagnii':'Pava andina','Podilymbus podiceps':'Zambullidor piquipinto','Tachybaptus dominicus':'Zambullidor menor','Tigrisoma lineatum':'Garza tigre rayada'}

@st.cache_resource
def download_model_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        try:
            with st.spinner(f'Descargando {output_path} desde Google Drive...'):
                gdown.download(f'https://drive.google.com/uc?id={file_id}', output_path, quiet=False)
            st.success(f'Descargado: {output_path}')
        except Exception as e:
            st.error(f'Error al descargar {output_path}: {e}'); return False
    return True

@st.cache_resource
def load_models():
    try:
        if not download_model_from_drive(GDRIVE_MODEL_1_ID, MODEL_1_PATH): return None
        if not download_model_from_drive(GDRIVE_MODEL_2_ID, MODEL_2_PATH): return None
        m1 = tf.keras.models.load_model(MODEL_1_PATH, compile=False)
        m2 = tf.keras.models.load_model(MODEL_2_PATH, compile=False)
        m1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        m2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return {"Modelo 1 (Inception)": m1, "Modelo 2 (VGG16)": m2}
    except Exception as e:
        st.error(f"Error al cargar los modelos: {e}")
        st.info("Verifica que:")
        st.write("- Los IDs de Google Drive sean correctos")
        st.write("- Los archivos esten compartidos como 'Cualquiera con el enlace'")
        st.write(f"- ID Modelo 1: {GDRIVE_MODEL_1_ID}")
        st.write(f"- ID Modelo 2: {GDRIVE_MODEL_2_ID}")
        return None

def predict(image, model_key, models_dict):
    model   = models_dict[model_key]
    classes = CLASSES_MODEL_1 if "Modelo 1" in model_key else CLASSES_MODEL_2
    img     = image.convert('RGB').resize(IMG_SIZE)
    arr     = np.expand_dims(np.array(img)/255.0, axis=0)
    preds   = model.predict(arr)
    idx     = int(np.argmax(preds, axis=1)[0])
    return classes[idx], float(preds[0][idx])*100, preds[0]

# ═══════════════════════════════════════════════════════════════════════════════
# PERSISTENCIA DE SESION
# ═══════════════════════════════════════════════════════════════════════════════
SESSION_FILE    = ".session_store.json"
INACTIVITY_MINS = 15

def save_session(email, name, role):
    now = datetime.now().isoformat()
    with open(SESSION_FILE,"w") as f:
        json.dump({"email":email,"name":name,"role":role,"last_active":now,"login_time":now}, f)

def load_session():
    if not os.path.exists(SESSION_FILE): return None
    try:
        with open(SESSION_FILE) as f: data = json.load(f)
        if datetime.now()-datetime.fromisoformat(data["last_active"]) > timedelta(minutes=INACTIVITY_MINS):
            clear_session(); return None
        if "login_time" not in data: data["login_time"] = data["last_active"]
        return data
    except Exception: return None

def update_session_activity():
    if not os.path.exists(SESSION_FILE): return
    try:
        with open(SESSION_FILE) as f: data = json.load(f)
        data["last_active"] = datetime.now().isoformat()
        with open(SESSION_FILE,"w") as f: json.dump(data, f)
    except Exception: pass

def clear_session():
    if os.path.exists(SESSION_FILE): os.remove(SESSION_FILE)

def seconds_until_timeout():
    if not os.path.exists(SESSION_FILE): return 0
    try:
        with open(SESSION_FILE) as f: data = json.load(f)
        return max(0, INACTIVITY_MINS*60 - (datetime.now()-datetime.fromisoformat(data["last_active"])).total_seconds())
    except Exception: return 0

def format_connected_time():
    ls = st.session_state.get("login_time")
    if not ls: return "—"
    try:
        d = datetime.now()-datetime.fromisoformat(ls)
        t = int(d.total_seconds())
        h,r = divmod(t,3600); m,s = divmod(r,60)
        if h>0: return f"{h}h {m:02d}m"
        elif m>0: return f"{m}m {s:02d}s"
        else: return f"{s}s"
    except Exception: return "—"

# ═══════════════════════════════════════════════════════════════════════════════
# PANTALLA DE AUTENTICACION
# ═══════════════════════════════════════════════════════════════════════════════
def auth_screen():
    # El CSS para ocultar el sidebar en auth ya está en static/styles.css
    # No se necesita inyección adicional aquí

    col_l, col_c, col_r = st.columns([1,2,1])
    with col_c:
        st.markdown("""
        <div style="text-align:center;margin-bottom:2.5rem;padding-top:2rem;">
            <span class="bird-icon">🦅</span>
            <div class="auth-logo">AvisFauna</div>
            <div class="auth-tagline">Identificacion inteligente de aves colombianas</div>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_register, tab_reset = st.tabs(["  Iniciar sesion  ","  Registrarse  ","  Olvide mi contrasena  "])

        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form("form_login"):
                email    = st.text_input("Correo electronico", placeholder="usuario@correo.com")
                password = st.text_input("Contrasena", type="password", placeholder="••••••••")
                submitted = st.form_submit_button("Entrar →", use_container_width=True)
            if submitted:
                if not email or not password:
                    st.error("Completa todos los campos.")
                else:
                    db_ensure_admin()
                    user = db_get_user(email)
                    if user and user["password"] == hash_password(password):
                        role = get_user_role(email)
                        st.session_state.authenticated = True
                        st.session_state.user_email    = email
                        st.session_state.user_name     = user["name"]
                        st.session_state.user_role     = role
                        st.session_state.login_time    = datetime.now().isoformat()
                        st.session_state.current_page  = "clasificador"
                        db_update_last_login(email)
                        save_session(email, user["name"], role)
                        st.success(f"Bienvenido/a, {user['name']}!")
                        st.rerun()
                    else:
                        st.error("Correo o contrasena incorrectos.")

        with tab_register:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form("form_register"):
                name      = st.text_input("Nombre completo", placeholder="Ana Garcia")
                reg_email = st.text_input("Correo electronico", placeholder="usuario@correo.com")
                reg_pass  = st.text_input("Contrasena", type="password", placeholder="Minimo 8 caracteres")
                reg_pass2 = st.text_input("Confirmar contrasena", type="password", placeholder="Repite tu contrasena")
                submitted_r = st.form_submit_button("Crear cuenta →", use_container_width=True)
            if submitted_r:
                ok = True
                if not name or not reg_email or not reg_pass or not reg_pass2:
                    st.error("Completa todos los campos."); ok=False
                elif not validate_email(reg_email):
                    st.error("Correo invalido."); ok=False
                elif db_get_user(reg_email):
                    st.error("Este correo ya esta registrado."); ok=False
                elif len(reg_pass)<8:
                    st.error("Contrasena de minimo 8 caracteres."); ok=False
                elif reg_pass!=reg_pass2:
                    st.error("Las contrasenas no coinciden."); ok=False
                if ok:
                    role = "administrador" if reg_email==ADMIN_EMAIL else "usuario"
                    db_create_user(reg_email, name, hash_password(reg_pass), role, datetime.now().isoformat())
                    st.success("Cuenta creada. Ya puedes iniciar sesion!")

        with tab_reset:
            st.markdown("<br>", unsafe_allow_html=True)
            if "reset_step" not in st.session_state: st.session_state.reset_step=1

            if st.session_state.reset_step==1:
                st.markdown('<div style="text-align:center;margin-bottom:1.2rem;"><div style="font-size:2rem;">🔐</div><div class="display-subtitle">Ingresa tu correo para recibir un codigo de recuperacion.</div></div>', unsafe_allow_html=True)
                with st.form("form_reset_req"):
                    reset_email = st.text_input("Correo electronico", placeholder="usuario@correo.com")
                    sub_req = st.form_submit_button("📨 Enviar codigo", use_container_width=True)
                if sub_req:
                    user = db_get_user(reset_email) if reset_email else None
                    if not reset_email: st.error("Ingresa tu correo.")
                    elif not validate_email(reset_email): st.error("Correo invalido.")
                    elif not user: st.error("No existe cuenta con ese correo.")
                    else:
                        token   = generate_token(8).upper()
                        expires = (datetime.now()+timedelta(minutes=15)).isoformat()
                        db_set_reset_token(reset_email, token, expires)
                        st.session_state.reset_email_target = reset_email
                        sent, error = send_reset_email(reset_email, user["name"], token)
                        if sent:
                            st.session_state.reset_step=2; st.session_state.reset_email_sent=True; st.rerun()
                        elif error=="no_config":
                            st.session_state.reset_step=2; st.session_state.reset_email_sent=False; st.session_state.reset_demo_token=token; st.rerun()
                        elif error=="auth_error": st.error("Error SMTP. Verifica credenciales.")
                        else: st.error(f"No se pudo enviar: {error}")

            elif st.session_state.reset_step==2:
                target_email = st.session_state.get("reset_email_target","")
                email_sent   = st.session_state.get("reset_email_sent",False)
                demo_token   = st.session_state.get("reset_demo_token","")
                if email_sent:
                    st.markdown(f'<div style="background:rgba(134,239,172,0.08);border:1px solid rgba(134,239,172,0.25);border-radius:12px;padding:1rem;margin-bottom:1.2rem;text-align:center;"><div style="font-size:1.5rem;">📬</div><div style="color:#86efac;font-weight:500;">Codigo enviado a {target_email}</div><div style="font-size:0.8rem;color:rgba(255,255,255,0.5);">Revisa spam si no llega</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="background:rgba(234,179,8,0.08);border:1px solid rgba(234,179,8,0.25);border-radius:12px;padding:0.9rem;margin-bottom:0.8rem;text-align:center;"><div style="color:#fde68a;font-weight:500;">⚙️ Modo demo — SMTP no configurado</div></div>', unsafe_allow_html=True)
                    if demo_token:
                        st.markdown(f'<div style="background:rgba(37,99,235,0.12);border:2px solid rgba(96,165,250,0.30);border-radius:12px;padding:1.2rem;text-align:center;margin-bottom:1rem;"><div style="font-size:0.73rem;color:rgba(255,255,255,0.42);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">Tu codigo</div><div style="font-size:2rem;font-weight:700;color:#60a5fa;letter-spacing:0.25em;font-family:monospace;">{demo_token}</div><div style="font-size:0.75rem;color:rgba(255,255,255,0.38);margin-top:6px;">Valido 15 minutos</div></div>', unsafe_allow_html=True)
                with st.form("form_reset_verify"):
                    input_token = st.text_input("Codigo de verificacion", placeholder="Ej: A3BX7K2M")
                    new_pass    = st.text_input("Nueva contrasena", type="password", placeholder="Minimo 8 caracteres")
                    new_pass2   = st.text_input("Confirmar contrasena", type="password", placeholder="Repite")
                    sub_verify  = st.form_submit_button("🔒 Restablecer contrasena", use_container_width=True)
                col_b,_ = st.columns([1,3])
                with col_b:
                    if st.button("← Volver", use_container_width=True):
                        for k in ["reset_step","reset_email_target","reset_email_sent","reset_demo_token"]: st.session_state.pop(k,None)
                        st.rerun()
                if sub_verify:
                    t_data = db_get_reset_token(target_email)
                    ok = True
                    if not input_token or not new_pass or not new_pass2: st.error("Completa todos los campos."); ok=False
                    elif not t_data: st.error("No hay codigo activo."); ok=False
                    elif datetime.now()>datetime.fromisoformat(t_data["expires"]): st.error("Codigo expirado."); ok=False
                    elif input_token.strip().upper()!=t_data["token"]: st.error("Codigo incorrecto."); ok=False
                    if ok:
                        if len(new_pass)<8: st.error("Minimo 8 caracteres.")
                        elif new_pass!=new_pass2: st.error("Las contrasenas no coinciden.")
                        else:
                            db_update_password(target_email, hash_password(new_pass))
                            db_delete_reset_token(target_email)
                            for k in ["reset_step","reset_email_target","reset_email_sent","reset_demo_token"]: st.session_state.pop(k,None)
                            st.success("Contrasena restablecida. Ya puedes iniciar sesion!")
                            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
def render_sidebar(models_dict):
    role    = st.session_state.get("user_role","usuario")
    email   = st.session_state.get("user_email","")
    name    = st.session_state.get("user_name","")
    current = st.session_state.get("current_page","clasificador")
    av_col  = avatar_color(role)
    t_conn  = format_connected_time()

    # CSS del sidebar cargado globalmente desde static/styles.css
    with st.sidebar:
        # Logo
        st.markdown("""
        <div style="padding:1.2rem 0 1rem;">
            <div style="font-family:'Playfair Display',serif;font-size:1.4rem;font-weight:700;color:#fff;">🦅 AvisFauna</div>
            <div style="font-size:0.73rem;color:rgba(255,255,255,0.32);margin-top:2px;">Identificacion inteligente</div>
        </div>
        <div class="sidebar-divider"></div>
        """, unsafe_allow_html=True)

        # Tarjeta de usuario
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.10);
                    border-radius:14px;padding:1rem;margin-bottom:0.8rem;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.85rem;">
                <div style="width:42px;height:42px;border-radius:50%;background:{av_col};
                            display:flex;align-items:center;justify-content:center;
                            font-weight:700;font-size:1.05rem;color:#fff;flex-shrink:0;
                            box-shadow:0 0 0 3px rgba(255,255,255,0.08);">
                    {name[0].upper() if name else "?"}
                </div>
                <div style="min-width:0;flex:1;">
                    <div style="color:#fff;font-size:0.9rem;font-weight:600;
                                white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{name}</div>
                    <div style="font-size:0.71rem;color:rgba(255,255,255,0.38);
                                white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{email}</div>
                </div>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:5px 0;border-top:1px solid rgba(255,255,255,0.07);">
                <span style="font-size:0.69rem;color:rgba(255,255,255,0.38);
                             text-transform:uppercase;letter-spacing:0.07em;">Rol</span>
                {role_badge_html(role)}
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:5px 0;border-top:1px solid rgba(255,255,255,0.07);">
                <span style="font-size:0.69rem;color:rgba(255,255,255,0.38);
                             text-transform:uppercase;letter-spacing:0.07em;">Conectado</span>
                <span style="font-size:0.82rem;font-weight:600;color:#60a5fa;">⏱ {t_conn}</span>
            </div>
        </div>
        <div class="sidebar-divider"></div>
        """, unsafe_allow_html=True)

        # Navegacion con st.radio (unico metodo confiable en Streamlit sidebar)
        st.markdown('<div class="label-muted" style="margin-bottom:0.5rem;">Navegacion</div>', unsafe_allow_html=True)

        nav_options = {"🔬 Clasificador": "clasificador"}
        if can(role,"ver_estadisticas"):   nav_options["📊 Mis estadisticas"]  = "estadisticas"
        if can(role,"dashboard_admin"):    nav_options["🛡️ Panel de admin"]    = "dashboard"
        if can(role,"gestionar_usuarios"): nav_options["👥 Gestion de roles"]  = "roles"

        labels  = list(nav_options.keys())
        ids     = list(nav_options.values())
        # Obtener index del page actual
        try:
            current_idx = ids.index(current)
        except ValueError:
            current_idx = 0

        selected_label = st.radio(
            "nav",
            options=labels,
            index=current_idx,
            label_visibility="collapsed"
        )
        selected_page = nav_options[selected_label]
        if selected_page != current:
            st.session_state.current_page = selected_page
            st.rerun()

        # Modelo
        st.markdown('<div class="sidebar-divider"></div><div class="label-muted" style="margin-bottom:0.4rem;">Modelo de clasificacion</div>', unsafe_allow_html=True)
        model_choice = st.selectbox("Modelo", list(models_dict.keys()), label_visibility="collapsed")

        # Mini estadisticas de aves
        st.markdown('<div class="sidebar-divider"></div><div class="label-muted" style="margin-bottom:0.6rem;">Mis aves mas examinadas</div>', unsafe_allow_html=True)
        top     = db_top_species(user_email=email, limit=5)
        total_u = db_count_predictions(user_email=email)

        if total_u == 0:
            st.markdown('<div style="font-size:0.78rem;color:rgba(255,255,255,0.3);text-align:center;padding:0.8rem 0;font-style:italic;">Sin clasificaciones aun</div>', unsafe_allow_html=True)
        else:
            max_cnt = top[0][1] if top else 1
            for sp, cnt in top:
                common  = COMMON_NAMES.get(sp,"")
                pct_bar = cnt / max_cnt * 100
                pct_tot = cnt / total_u * 100
                st.markdown(f"""
                <div style="margin-bottom:9px;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                        <div style="min-width:0;flex:1;">
                            <div style="font-size:0.76rem;font-style:italic;color:rgba(255,255,255,0.72);
                                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{sp}</div>
                            <div style="font-size:0.67rem;color:rgba(255,255,255,0.35);">{common}</div>
                        </div>
                        <div style="text-align:right;flex-shrink:0;margin-left:6px;">
                            <div style="font-size:0.8rem;font-weight:700;color:#93c5fd;">{cnt}x</div>
                            <div style="font-size:0.67rem;color:rgba(255,255,255,0.3);">{pct_tot:.0f}%</div>
                        </div>
                    </div>
                    <div style="background:rgba(255,255,255,0.06);border-radius:3px;height:4px;">
                        <div style="width:{pct_bar:.1f}%;height:4px;border-radius:3px;
                                    background:linear-gradient(90deg,#1d4ed8,#60a5fa);"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:0.7rem;color:rgba(255,255,255,0.3);text-align:right;margin-top:2px;">{total_u} clasificaciones totales</div>', unsafe_allow_html=True)

        # Cerrar sesion
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        if st.button("🚪 Cerrar sesion", use_container_width=True):
            clear_session()
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()

    return model_choice

# ═══════════════════════════════════════════════════════════════════════════════
# PAGINA: CLASIFICADOR
# ═══════════════════════════════════════════════════════════════════════════════
def page_clasificador(models_dict, model_choice):
    col_h1,col_h2 = st.columns([2,1])
    with col_h1:
        st.markdown(f'<div class="animate-in"><div class="display-title">Identificador<br>de Aves 🦅</div><div class="display-subtitle" style="margin-top:0.7rem;max-width:500px;">Sube una fotografia y nuestro modelo identificara la especie con alta precision.</div></div>', unsafe_allow_html=True)
    with col_h2:
        st.markdown(f'<div style="text-align:right;padding-top:1rem;"><div class="model-chip"><span>🧠</span> {model_choice}</div><div style="font-size:0.8rem;color:rgba(255,255,255,0.35);margin-top:8px;">{len(CLASSES_MODEL_1)} especies · Colombia</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col_upload,col_result = st.columns([1,1], gap="large")
    with col_upload:
        st.markdown('<div class="card animate-in">', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="margin-bottom:1.2rem;">📤 Sube tu imagen</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Elige un archivo", type=["jpg","png","jpeg"], label_visibility="collapsed")
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen cargada", use_column_width=True, output_format="PNG")
            st.markdown(f'<div style="font-size:0.78rem;color:rgba(255,255,255,0.4);margin-top:0.6rem;">{image.width}x{image.height}px · {uploaded_file.name}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align:center;padding:2.5rem 1rem;color:rgba(255,255,255,0.3);"><div style="font-size:2.5rem;margin-bottom:0.8rem;">🖼️</div><div style="font-size:0.9rem;">Arrastra o selecciona una imagen JPG / PNG</div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="card animate-in" style="min-height:320px;">', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="margin-bottom:1.2rem;">🔬 Resultado</div>', unsafe_allow_html=True)
        if uploaded_file:
            if st.button("✦ Identificar especie", use_container_width=True):
                with st.spinner("Analizando imagen..."):
                    try:
                        species, confidence, all_preds = predict(image, model_choice, models_dict)
                        common = COMMON_NAMES.get(species,"—")
                        active_classes = CLASSES_MODEL_1 if "Modelo 1" in model_choice else CLASSES_MODEL_2
                        st.session_state.last_result = {"species":species,"common":common,"confidence":confidence,"all_preds":all_preds.tolist(),"classes":active_classes}
                        db_record_prediction(st.session_state.user_email, species, confidence, model_choice)
                    except Exception as e: st.error(f"Error: {e}")
        if "last_result" in st.session_state and uploaded_file:
            r = st.session_state.last_result
            st.markdown(f'<div style="margin-bottom:1.5rem;"><div class="result-label">Especie identificada</div><div class="result-species">{r["species"]}</div><div style="font-size:0.9rem;color:rgba(255,255,255,0.5);margin-top:2px;">{r["common"]}</div></div><div style="margin-bottom:1.5rem;"><div class="result-label">Confianza</div><div class="result-confidence">{r["confidence"]:.1f}%</div></div>', unsafe_allow_html=True)
            st.progress(r["confidence"]/100)
        else:
            st.markdown('<div style="text-align:center;padding:3rem 1rem;color:rgba(255,255,255,0.25);"><div style="font-size:2rem;margin-bottom:0.5rem;">🔍</div><div style="font-size:0.9rem;">Sube una imagen y pulsa Identificar especie</div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if "last_result" in st.session_state and uploaded_file:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card animate-in">', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="margin-bottom:1.4rem;">📊 Distribucion de probabilidades</div>', unsafe_allow_html=True)
        r = st.session_state.last_result
        sorted_pairs = sorted(zip(r["classes"],r["all_preds"]), key=lambda x:x[1], reverse=True)
        cols = st.columns(2)
        for i,(cls,prob) in enumerate(sorted_pairs):
            pct=prob*100; common=COMMON_NAMES.get(cls,""); is_top=cls==r["species"]; color="#60a5fa" if is_top else "#475569"
            with cols[i%2]:
                st.markdown(f'<div style="margin-bottom:14px;"><div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:5px;"><div><span style="font-size:0.82rem;font-style:italic;color:{"#60a5fa" if is_top else "rgba(255,255,255,0.65)"};">{cls}</span><span style="font-size:0.7rem;color:rgba(255,255,255,0.35);margin-left:6px;">{common}</span></div><span style="font-size:0.82rem;font-weight:600;color:{"#93c5fd" if is_top else "rgba(255,255,255,0.5)"};">{pct:.1f}%</span></div><div class="prob-bar-container"><div class="prob-bar-fill" style="width:{pct:.1f}%;background:{color};"></div></div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGINA: MIS ESTADISTICAS
# ═══════════════════════════════════════════════════════════════════════════════
def page_estadisticas():
    email    = st.session_state.user_email
    total    = db_count_predictions(email)
    avg_conf = db_avg_confidence(email)
    top      = db_top_species(email, limit=10)
    top_sp   = top[0][0] if top else "—"
    top_cmn  = COMMON_NAMES.get(top_sp, top_sp)

    st.markdown('<div class="page-title animate-in">📊 Mis estadisticas</div>', unsafe_allow_html=True)
    st.markdown('<div class="display-subtitle" style="margin-bottom:2rem;">Tu actividad de clasificacion en AvisFauna</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,(icon,val,label) in zip([c1,c2,c3,c4],[("🔬",str(total),"Clasificaciones"),("🎯",f"{avg_conf}%","Confianza media"),("🦅",top_cmn[:16],"Ave mas vista"),("⏱",format_connected_time(),"Tiempo sesion")]):
        with col: st.markdown(f'<div class="stat-card"><div class="stat-icon">{icon}</div><div class="stat-value">{val}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if total==0:
        st.markdown('<div class="alert-info">Aun no has realizado ninguna clasificacion.</div>', unsafe_allow_html=True); return

    col_a,col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="margin-bottom:1.2rem;">🦅 Especies clasificadas</div>', unsafe_allow_html=True)
        for sp,cnt in top:
            pct=cnt/total*100; common=COMMON_NAMES.get(sp,"")
            st.markdown(f'<div style="margin-bottom:12px;"><div style="display:flex;justify-content:space-between;margin-bottom:4px;"><div><span style="font-size:0.83rem;font-style:italic;color:rgba(255,255,255,0.75);">{sp}</span><span style="font-size:0.72rem;color:rgba(255,255,255,0.35);margin-left:6px;">{common}</span></div><span style="font-size:0.83rem;color:#93c5fd;font-weight:600;">{cnt}x</span></div><div class="prob-bar-container"><div class="prob-bar-fill" style="width:{pct:.1f}%;background:#2563eb;"></div></div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="margin-bottom:1.2rem;">🕘 Actividad reciente</div>', unsafe_allow_html=True)
        for p in db_get_predictions(user_email=email, limit=8):
            ts=datetime.fromisoformat(p["timestamp"]).strftime("%d %b %Y · %H:%M")
            conf_c="#86efac" if p["confidence"]>=80 else "#fde68a" if p["confidence"]>=60 else "#fca5a5"
            st.markdown(f'<div style="display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.06);"><div><div style="font-size:0.83rem;color:rgba(255,255,255,0.8);font-style:italic;">{p["species"]}</div><div style="font-size:0.73rem;color:rgba(255,255,255,0.35);">{ts}</div></div><span style="font-size:0.82rem;font-weight:600;color:{conf_c};">{p["confidence"]:.1f}%</span></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGINA: PANEL DE ADMINISTRACION
# ═══════════════════════════════════════════════════════════════════════════════
def page_dashboard_admin():
    users    = db_get_all_users()
    user_map = {u["email"]:u for u in users}
    total_u  = len(users)
    total_p  = db_count_predictions()
    avg_conf = db_avg_confidence()
    roles_cnt= Counter(u["role"] for u in users)
    today_str= datetime.now().date().isoformat()
    active_today = len(set(p["user_email"] for p in db_get_predictions() if p["timestamp"][:10]==today_str))

    st.markdown('<div class="page-title animate-in">🛡️ Panel de administracion</div>', unsafe_allow_html=True)
    st.markdown('<div class="display-subtitle" style="margin-bottom:2rem;">Estadisticas globales del sistema AvisFauna</div>', unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(icon,val,label) in zip([c1,c2,c3,c4,c5],[("👥",str(total_u),"Usuarios"),("👑",str(roles_cnt.get("administrador",0)),"Admins"),("🔬",str(roles_cnt.get("experto",0)),"Expertos"),("🖼️",str(total_p),"Clasificaciones"),("⚡",str(active_today),"Activos hoy")]):
        with col: st.markdown(f'<div class="stat-card"><div class="stat-icon">{icon}</div><div class="stat-value">{val}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    tab_stats,tab_species = st.tabs(["📈 Estadisticas globales","🦅 Por especie"])
    with tab_stats:
        st.markdown("<br>", unsafe_allow_html=True)
        if total_p==0: st.markdown('<div class="alert-info">Aun no hay clasificaciones.</div>', unsafe_allow_html=True)
        else:
            col_a,col_b = st.columns(2)
            with col_a:
                st.markdown('<div class="card"><div class="section-title" style="margin-bottom:1.2rem;">👤 Clasificaciones por usuario</div>', unsafe_allow_html=True)
                user_cnt = Counter(p["user_email"] for p in db_get_predictions()); max_c=max(user_cnt.values()) if user_cnt else 1
                for em,cnt in user_cnt.most_common(10):
                    nm=user_map.get(em,{}).get("name",em.split("@")[0]); rl=user_map.get(em,{}).get("role","usuario"); pct=cnt/max_c*100
                    st.markdown(f'<div style="margin-bottom:11px;"><div style="display:flex;justify-content:space-between;margin-bottom:4px;flex-wrap:wrap;gap:4px;"><span style="font-size:0.82rem;color:rgba(255,255,255,0.75);">{nm} {role_badge_html(rl)}</span><span style="font-size:0.82rem;color:#93c5fd;font-weight:600;">{cnt}</span></div><div class="prob-bar-container"><div class="prob-bar-fill" style="width:{pct:.1f}%;background:#2563eb;"></div></div></div>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with col_b:
                st.markdown('<div class="card"><div class="section-title" style="margin-bottom:1.2rem;">🧠 Uso por modelo</div>', unsafe_allow_html=True)
                model_cnt=Counter(p["model"] for p in db_get_predictions())
                for model,cnt in model_cnt.most_common():
                    pct=cnt/total_p*100
                    st.markdown(f'<div style="margin-bottom:12px;"><div style="display:flex;justify-content:space-between;margin-bottom:4px;"><span style="font-size:0.83rem;color:rgba(255,255,255,0.75);">{model}</span><span style="font-size:0.83rem;color:#93c5fd;font-weight:600;">{cnt} · {pct:.0f}%</span></div><div class="prob-bar-container"><div class="prob-bar-fill" style="width:{pct:.1f}%;background:#7c3aed;"></div></div></div>', unsafe_allow_html=True)
                st.markdown(f'<div style="text-align:center;padding:1rem 0;"><div class="stat-label">Confianza promedio global</div><div style="font-size:2.5rem;font-weight:700;color:#60a5fa;">{avg_conf}%</div><div class="stat-label">sobre {total_p} clasificaciones</div></div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="card"><div class="section-title" style="margin-bottom:1.2rem;">🕘 Ultimas clasificaciones</div>', unsafe_allow_html=True)
            hcols=st.columns([2,2,1,1,2])
            for hc,lbl in zip(hcols,["Usuario","Especie","Confianza","Modelo","Fecha"]):
                with hc: st.markdown(f'<div class="label-muted">{lbl}</div>', unsafe_allow_html=True)
            for p in db_get_predictions(limit=12):
                nm=user_map.get(p["user_email"],{}).get("name",p["user_email"].split("@")[0]); ts=datetime.fromisoformat(p["timestamp"]).strftime("%d %b · %H:%M"); conf_c="#86efac" if p["confidence"]>=80 else "#fde68a" if p["confidence"]>=60 else "#fca5a5"; mod_s="Inception" if "Modelo 1" in p["model"] else "VGG16"
                for rc,html in zip(st.columns([2,2,1,1,2]),[f'<span style="font-size:0.82rem;color:rgba(255,255,255,0.75);">{nm}</span>',f'<span style="font-size:0.82rem;font-style:italic;color:rgba(255,255,255,0.65);">{p["species"]}</span>',f'<span style="font-size:0.82rem;font-weight:600;color:{conf_c};">{p["confidence"]:.1f}%</span>',f'<span style="font-size:0.75rem;color:rgba(255,255,255,0.4);">{mod_s}</span>',f'<span style="font-size:0.75rem;color:rgba(255,255,255,0.4);">{ts}</span>']):
                    with rc: st.markdown(f'<div style="padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.05);">{html}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab_species:
        st.markdown("<br>", unsafe_allow_html=True)
        top_all=db_top_species(limit=10)
        if not top_all: st.markdown('<div class="alert-info">Sin clasificaciones.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="card"><div class="section-title" style="margin-bottom:1.4rem;">🦅 Frecuencia por especie</div>', unsafe_allow_html=True)
            max_sp=top_all[0][1]
            for sp,cnt in top_all:
                common=COMMON_NAMES.get(sp,""); pct_bar=cnt/max_sp*100; pct_tot=cnt/total_p*100
                st.markdown(f'<div style="margin-bottom:14px;padding:1rem;background:rgba(255,255,255,0.03);border-radius:10px;border:1px solid rgba(255,255,255,0.07);"><div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;flex-wrap:wrap;gap:8px;"><div><span style="font-size:0.87rem;font-style:italic;color:#93c5fd;">{sp}</span><span style="font-size:0.75rem;color:rgba(255,255,255,0.38);margin-left:8px;">{common}</span></div><span style="font-size:0.8rem;color:rgba(255,255,255,0.5);">{cnt} veces · {pct_tot:.1f}% del total</span></div><div class="prob-bar-container"><div class="prob-bar-fill" style="width:{pct_bar:.1f}%;background:linear-gradient(90deg,#1d4ed8,#60a5fa);"></div></div></div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGINA: GESTION DE ROLES
# ═══════════════════════════════════════════════════════════════════════════════
def page_roles():
    users     = db_get_all_users()
    total_u   = len(users)
    roles_cnt = Counter(u["role"] for u in users)
    colors_r  = {"administrador":"#fca5a5","experto":"#86efac","usuario":"#93c5fd"}

    st.markdown('<div class="page-title animate-in">👥 Gestion de roles</div>', unsafe_allow_html=True)
    st.markdown('<div class="display-subtitle" style="margin-bottom:1.5rem;">Administra los roles y permisos de todos los usuarios registrados</div>', unsafe_allow_html=True)

    # Cards resumen
    c1,c2,c3 = st.columns(3)
    for col,r in zip([c1,c2,c3],["administrador","experto","usuario"]):
        ri=ROLES[r]; cnt=roles_cnt.get(r,0); pct=cnt/total_u*100 if total_u else 0
        with col: st.markdown(f'<div class="stat-card" style="margin-bottom:1rem;"><div class="stat-icon">{ri["icon"]}</div><div class="stat-value">{cnt}</div><div class="stat-label">{ri["label"]}s</div><div style="background:rgba(255,255,255,0.07);border-radius:3px;height:4px;margin-top:10px;"><div style="width:{pct:.1f}%;height:4px;border-radius:3px;background:{colors_r[r]};opacity:0.7;"></div></div><div style="font-size:0.7rem;color:rgba(255,255,255,0.3);margin-top:4px;">{pct:.0f}% del total</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Filtros
    col_s,col_f = st.columns([2,1])
    with col_s: search = st.text_input("🔍 Buscar usuario", placeholder="Nombre o correo...", key="roles_search")
    with col_f: rol_filter = st.selectbox("Filtrar por rol", ["Todos","Administrador","Experto","Usuario"])
    rol_map = {"Todos":None,"Administrador":"administrador","Experto":"experto","Usuario":"usuario"}
    filter_role = rol_map[rol_filter]

    st.markdown("<br>", unsafe_allow_html=True)

    # Cabecera tabla
    st.markdown('<div style="display:grid;grid-template-columns:2fr 2fr 1fr 0.8fr 1.2fr;gap:8px;padding:8px 14px;margin-bottom:4px;"><div class="label-muted">Usuario</div><div class="label-muted">Correo</div><div class="label-muted">Rol actual</div><div class="label-muted">Clasif.</div><div class="label-muted">Registro</div></div>', unsafe_allow_html=True)

    any_shown = False
    for u in users:
        email_u = u["email"]; name_u=u["name"]; role_u=u["role"]
        created = u.get("created_at","")[:10] if u.get("created_at") else "—"
        last_login = u.get("last_login","")[:10] if u.get("last_login") else "Nunca"
        user_preds = db_count_predictions(email_u)
        is_fixed   = (email_u==ADMIN_EMAIL)
        if search and search.lower() not in name_u.lower() and search.lower() not in email_u.lower(): continue
        if filter_role and role_u!=filter_role: continue
        any_shown = True

        st.markdown(f'<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:0.9rem 1rem;margin-bottom:4px;">', unsafe_allow_html=True)
        col_n,col_e,col_r,col_p,col_d = st.columns([2,2,1,0.8,1.2])
        with col_n:
            av=avatar_color(role_u)
            st.markdown(f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0;"><div style="width:32px;height:32px;border-radius:50%;background:{av};display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.85rem;color:#fff;flex-shrink:0;">{name_u[0].upper()}</div><span style="font-size:0.85rem;color:rgba(255,255,255,0.85);font-weight:500;">{name_u}</span></div>', unsafe_allow_html=True)
        with col_e: st.markdown(f'<div style="font-size:0.77rem;color:rgba(255,255,255,0.42);padding:8px 0;word-break:break-all;">{email_u}</div>', unsafe_allow_html=True)
        with col_r: st.markdown(f'<div style="padding:8px 0;">{role_badge_html(role_u)}</div>', unsafe_allow_html=True)
        with col_p: st.markdown(f'<div style="font-size:0.9rem;font-weight:700;color:#93c5fd;padding:10px 0;">{user_preds}</div>', unsafe_allow_html=True)
        with col_d: st.markdown(f'<div style="font-size:0.78rem;color:rgba(255,255,255,0.45);padding:10px 0;">{created}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if is_fixed:
            st.markdown('<div class="alert-warn" style="margin:-2px 0 8px;font-size:0.8rem;">👑 Administrador principal — rol permanente.</div>', unsafe_allow_html=True)
        else:
            col_sel,col_btn,col_info = st.columns([1.5,0.8,3])
            with col_sel:
                new_role = st.selectbox("Nuevo rol",list(ROLES.keys()),index=list(ROLES.keys()).index(role_u),key=f"rs_{email_u}",label_visibility="collapsed")
            with col_btn:
                if st.button("💾 Guardar",key=f"rb_{email_u}",use_container_width=True):
                    db_update_role(email_u, new_role)
                    st.success(f"✅ {name_u} → {ROLES[new_role]['label']}")
                    st.rerun()
            with col_info:
                st.markdown(f'<div style="font-size:0.75rem;color:rgba(255,255,255,0.35);padding:10px 0;">Ultimo acceso: {last_login}</div>', unsafe_allow_html=True)
        st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

    if not any_shown:
        st.markdown('<div class="alert-info">No se encontraron usuarios con ese filtro.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# APP PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
def main_app():
    db_ensure_admin()
    models_dict = load_models()
    if models_dict is None:
        st.error("No se pudieron cargar los modelos. Verifica Google Drive."); return
    model_choice = render_sidebar(models_dict)
    current_page = st.session_state.get("current_page","clasificador")
    role         = st.session_state.get("user_role","usuario")
    if   current_page=="clasificador": page_clasificador(models_dict, model_choice)
    elif current_page=="estadisticas":
        if can(role,"ver_estadisticas"): page_estadisticas()
        else: st.error("Sin permiso.")
    elif current_page=="dashboard":
        if can(role,"dashboard_admin"): page_dashboard_admin()
        else: st.error("Acceso restringido.")
    elif current_page=="roles":
        if can(role,"gestionar_usuarios"): page_roles()
        else: st.error("Acceso restringido.")

# ═══════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    init_db()
    if "authenticated" not in st.session_state:
        saved = load_session()
        if saved:
            st.session_state.authenticated = True
            st.session_state.user_email    = saved["email"]
            st.session_state.user_name     = saved["name"]
            st.session_state.user_role     = saved["role"]
            st.session_state.login_time    = saved.get("login_time", datetime.now().isoformat())
            st.session_state.current_page  = "clasificador"
        else:
            st.session_state.authenticated = False

    if st.session_state.get("authenticated"):
        secs_left = seconds_until_timeout()
        if secs_left<=0:
            clear_session()
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.session_state.authenticated=False; st.session_state.session_expired=True; st.rerun()
        else:
            update_session_activity()
            if secs_left<=180:
                mins=int(secs_left//60); secs=int(secs_left%60)
                st.markdown(f'<div style="position:fixed;bottom:1.2rem;right:1.4rem;z-index:9999;background:rgba(180,100,0,0.92);border:1px solid rgba(253,186,116,0.5);border-radius:12px;padding:10px 18px;display:flex;align-items:center;gap:10px;backdrop-filter:blur(8px);box-shadow:0 4px 20px rgba(0,0,0,0.4);"><span style="font-size:1.1rem;">⏱️</span><div><div style="font-size:0.78rem;color:rgba(255,255,255,0.65);line-height:1;">Sesion expira en</div><div style="font-size:1rem;font-weight:700;color:#fff;line-height:1.3;">{mins}:{secs:02d}</div></div></div>', unsafe_allow_html=True)

    if st.session_state.get("session_expired"):
        st.warning("Tu sesion expiro por inactividad. Inicia sesion de nuevo.")
        del st.session_state["session_expired"]

    if not st.session_state.get("authenticated"):
        auth_screen()
    else:
        main_app()

if __name__ == "__main__":
    main()