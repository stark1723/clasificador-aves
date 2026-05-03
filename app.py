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
from datetime import datetime, timedelta
from collections import Counter

# ═══════════════════════════════════════════════════════════════════════════════
# ─── CONFIGURACIÓN DE PÁGINA
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AvisFauna — Identificador de Aves",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════════════════════
# ─── ESTILOS GLOBALES
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a1628 0%, #0f2240 50%, #0d1f35 100%);
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* ── Sidebar oculto en auth ── */
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
.auth-mode [data-testid="stSidebarCollapsedControl"] { display: none !important; }

/* ── Tarjetas ── */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 2rem;
    backdrop-filter: blur(12px);
}
.card-elevated {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 20px;
    padding: 2.5rem;
    backdrop-filter: blur(16px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.stat-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    text-align: center;
}
.stat-value { font-size: 2rem; font-weight: 700; color: #ffffff; line-height: 1; margin-bottom: 4px; }
.stat-label { font-size: 0.73rem; color: rgba(255,255,255,0.42); text-transform: uppercase; letter-spacing: 0.08em; }
.stat-icon  { font-size: 1.4rem; margin-bottom: 8px; }

/* ── Tipografía ── */
.display-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem; font-weight: 700; color: #ffffff;
    line-height: 1.2; letter-spacing: -0.02em;
}
.display-subtitle { font-size: 1.05rem; color: rgba(255,255,255,0.55); font-weight: 300; }
.section-title { font-family: 'Playfair Display', serif; font-size: 1.4rem; font-weight: 600; color: #ffffff; }
.page-title   { font-family: 'Playfair Display', serif; font-size: 2rem;   font-weight: 700; color: #ffffff; margin-bottom: 0.3rem; }
.label-muted  { font-size: 0.78rem; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.08em; font-weight: 500; }

/* ── Role badges ── */
.role-admin   { display:inline-block; background:linear-gradient(135deg,#7c2d12,#b91c1c); color:#fca5a5; font-size:0.72rem; font-weight:600; padding:3px 10px; border-radius:20px; border:1px solid rgba(252,165,165,0.25); letter-spacing:0.04em; }
.role-experto { display:inline-block; background:linear-gradient(135deg,#14532d,#15803d); color:#86efac; font-size:0.72rem; font-weight:600; padding:3px 10px; border-radius:20px; border:1px solid rgba(134,239,172,0.25); letter-spacing:0.04em; }
.role-usuario { display:inline-block; background:linear-gradient(135deg,#1e3a5f,#1d4ed8); color:#93c5fd; font-size:0.72rem; font-weight:600; padding:3px 10px; border-radius:20px; border:1px solid rgba(147,197,253,0.25); letter-spacing:0.04em; }

/* ── Resultado ── */
.result-species    { font-family:'Playfair Display',serif; font-size:1.7rem; font-weight:600; color:#60a5fa; font-style:italic; margin:0.5rem 0; }
.result-confidence { font-size:2.5rem; font-weight:700; color:#ffffff; line-height:1; }
.result-label      { font-size:0.8rem; color:rgba(255,255,255,0.45); text-transform:uppercase; letter-spacing:0.08em; }

/* ── Barras de probabilidad ── */
.prob-bar-container { background:rgba(255,255,255,0.06); border-radius:4px; height:6px; width:100%; overflow:hidden; }
.prob-bar-fill      { height:100%; border-radius:4px; transition:width 0.8s ease; }

/* ── Auth ── */
.auth-container { max-width:480px; margin:2rem auto; }
.auth-logo      { font-family:'Playfair Display',serif; font-size:1.9rem; font-weight:700; color:#ffffff; text-align:center; margin-bottom:0.3rem; }
.auth-tagline   { text-align:center; color:rgba(255,255,255,0.45); font-size:0.9rem; margin-bottom:2rem; }
.bird-icon      { font-size:3rem; display:block; text-align:center; margin-bottom:0.5rem; filter:drop-shadow(0 0 20px rgba(96,165,250,0.4)); }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stSelectbox > div > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(96,165,250,0.6) !important;
    box-shadow: 0 0 0 3px rgba(96,165,250,0.12) !important;
}
.stTextInput label, .stSelectbox label {
    color: rgba(255,255,255,0.70) !important;
    font-size: 0.88rem !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Botones ── */
.stButton > button {
    background: linear-gradient(135deg,#1d4ed8,#2563eb) !important;
    color: #ffffff !important; border: none !important;
    border-radius: 10px !important; font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important; font-size: 0.95rem !important;
    padding: 0.65rem 1.5rem !important; transition: all 0.2s !important; width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#1e40af,#1d4ed8) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(37,99,235,0.35) !important;
}
.stButton > button:active { transform: translateY(0px) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(6,18,40,0.92) !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p { color: rgba(255,255,255,0.75) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(96,165,250,0.30) !important;
    border-radius: 14px !important;
    background: rgba(96,165,250,0.04) !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(96,165,250,0.55) !important;
    background: rgba(96,165,250,0.07) !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg,#2563eb,#60a5fa) !important;
    border-radius: 4px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 10px !important; padding: 4px !important; gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    color: rgba(255,255,255,0.55) !important;
    font-family: 'DM Sans', sans-serif !important; border-radius: 8px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(37,99,235,0.35) !important; color: #ffffff !important;
}

/* ── Model chip ── */
.model-chip {
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(96,165,250,0.12); border:1px solid rgba(96,165,250,0.25);
    color:#93c5fd; font-size:0.78rem; font-weight:500; padding:4px 10px; border-radius:6px;
}

/* ── Alertas ── */
.alert-warn { background:rgba(234,179,8,0.10); border:1px solid rgba(234,179,8,0.25); border-radius:10px; padding:0.8rem 1rem; color:#fde68a; font-size:0.88rem; }
.alert-info { background:rgba(96,165,250,0.08); border:1px solid rgba(96,165,250,0.20); border-radius:10px; padding:0.8rem 1rem; color:#93c5fd; font-size:0.88rem; }

/* ── Animación ── */
@keyframes fadeSlideIn { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:translateY(0); } }
.animate-in { animation: fadeSlideIn 0.5s ease forwards; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ─── ROLES Y PERMISOS
# ═══════════════════════════════════════════════════════════════════════════════

ADMIN_EMAIL = "adminfs01@gmail.com"

ROLES = {
    "administrador": {"label": "Administrador", "css": "role-admin",   "icon": "👑"},
    "experto":       {"label": "Experto",        "css": "role-experto", "icon": "🔬"},
    "usuario":       {"label": "Usuario",         "css": "role-usuario", "icon": "🙋"},
}

ROLE_PERMISSIONS = {
    "administrador": ["clasificar", "ver_estadisticas", "dashboard_admin", "gestionar_usuarios"],
    "experto":       ["clasificar", "ver_estadisticas"],
    "usuario":       ["clasificar"],
}

def can(role, permission):
    return permission in ROLE_PERMISSIONS.get(role, [])

def role_badge_html(role):
    r = ROLES.get(role, ROLES["usuario"])
    return f'<span class="{r["css"]}">{r["icon"]} {r["label"]}</span>'

def avatar_color(role):
    return {"administrador": "#b91c1c", "experto": "#15803d", "usuario": "#1d4ed8"}.get(role, "#1d4ed8")


# ═══════════════════════════════════════════════════════════════════════════════
# ─── BASE DE DATOS JSON LOCAL
# ═══════════════════════════════════════════════════════════════════════════════

USERS_FILE        = "users_db.json"
RESET_TOKENS_FILE = "reset_tokens.json"
STATS_FILE        = "stats_db.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def load_reset_tokens():
    if os.path.exists(RESET_TOKENS_FILE):
        with open(RESET_TOKENS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_reset_tokens(tokens):
    with open(RESET_TOKENS_FILE, "w") as f:
        json.dump(tokens, f, indent=2)

def load_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, "r") as f:
            return json.load(f)
    return {"predictions": []}

def save_stats(stats):
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

def record_prediction(user_email, species, confidence, model_used):
    stats = load_stats()
    stats["predictions"].append({
        "user_email": user_email,
        "species":    species,
        "confidence": round(confidence, 2),
        "model":      model_used,
        "timestamp":  datetime.now().isoformat()
    })
    save_stats(stats)

def ensure_admin_role():
    """Garantiza que adminfs01@gmail.com siempre sea administrador."""
    users = load_users()
    if ADMIN_EMAIL in users and users[ADMIN_EMAIL].get("role") != "administrador":
        users[ADMIN_EMAIL]["role"] = "administrador"
        save_users(users)

def get_user_role(email):
    if email == ADMIN_EMAIL:
        return "administrador"
    return load_users().get(email, {}).get("role", "usuario")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def validate_email(email):
    return re.match(r'^[\w\.-]+@[\w\.-]+\.\w{2,}$', email)

def generate_token(length=8):
    return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(length))


# ═══════════════════════════════════════════════════════════════════════════════
# ─── ENVÍO DE CORREO (SMTP)
# ═══════════════════════════════════════════════════════════════════════════════

def get_email_config():
    """
    Lee credenciales SMTP desde st.secrets.
    Configura en .streamlit/secrets.toml:
        [email]
        sender   = "tuemail@gmail.com"
        password = "tu_app_password_gmail"
    """
    try:
        sender   = st.secrets["email"]["sender"]
        password = st.secrets["email"]["password"]
        return sender, password
    except Exception:
        return None, None


def send_reset_email(dest_email, dest_name, token):
    """
    Envía el código de recuperación por correo usando SMTP/TLS (Gmail).
    Retorna (True, "") si tiene éxito o (False, codigo_error) si falla.
    """
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    sender, password = get_email_config()
    if not sender or not password:
        return False, "no_config"

    html_body = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head><meta charset="UTF-8"></head>
    <body style="margin:0;padding:0;background:#0a1628;font-family:'Segoe UI',Arial,sans-serif;">
      <table width="100%" cellpadding="0" cellspacing="0" style="background:#0a1628;padding:40px 0;">
        <tr><td align="center">
          <table width="520" cellpadding="0" cellspacing="0"
                 style="background:#0f1f38;border-radius:16px;border:1px solid rgba(255,255,255,0.1);overflow:hidden;">
            <tr>
              <td style="background:linear-gradient(135deg,#1d4ed8,#2563eb);padding:32px 40px;text-align:center;">
                <div style="font-size:2.2rem;">🦅</div>
                <div style="font-family:Georgia,serif;font-size:1.6rem;font-weight:700;color:#ffffff;margin-top:8px;">
                  AvisFauna
                </div>
                <div style="font-size:0.82rem;color:rgba(255,255,255,0.65);margin-top:4px;">
                  Recuperación de contraseña
                </div>
              </td>
            </tr>
            <tr>
              <td style="padding:36px 40px;">
                <p style="color:rgba(255,255,255,0.85);font-size:0.97rem;margin:0 0 16px 0;line-height:1.6;">
                  Hola <strong style="color:#ffffff;">{dest_name}</strong>,
                </p>
                <p style="color:rgba(255,255,255,0.65);font-size:0.92rem;margin:0 0 28px 0;line-height:1.6;">
                  Recibimos una solicitud para restablecer la contrasena de tu cuenta en AvisFauna.
                  Usa el siguiente codigo de verificacion:
                </p>
                <div style="background:rgba(37,99,235,0.15);border:2px solid rgba(96,165,250,0.35);
                            border-radius:12px;padding:24px;text-align:center;margin-bottom:28px;">
                  <div style="font-size:0.75rem;color:rgba(255,255,255,0.45);
                              text-transform:uppercase;letter-spacing:0.12em;margin-bottom:10px;">
                    Codigo de verificacion
                  </div>
                  <div style="font-size:2.4rem;font-weight:700;color:#60a5fa;
                              letter-spacing:0.3em;font-family:'Courier New',monospace;">
                    {token}
                  </div>
                  <div style="font-size:0.78rem;color:rgba(255,255,255,0.4);margin-top:10px;">
                    Valido durante <strong style="color:rgba(255,255,255,0.6);">15 minutos</strong>
                  </div>
                </div>
                <p style="color:rgba(255,255,255,0.4);font-size:0.83rem;line-height:1.6;margin:0;">
                  Si no solicitaste este cambio, puedes ignorar este correo. Tu cuenta permanece segura.
                </p>
              </td>
            </tr>
            <tr>
              <td style="background:rgba(0,0,0,0.2);padding:20px 40px;
                         border-top:1px solid rgba(255,255,255,0.07);text-align:center;">
                <p style="color:rgba(255,255,255,0.25);font-size:0.78rem;margin:0;">
                  AvisFauna &mdash; Identificacion inteligente de aves colombianas
                </p>
              </td>
            </tr>
          </table>
        </td></tr>
      </table>
    </body>
    </html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "AvisFauna - Codigo de recuperacion de contrasena"
    msg["From"]    = f"AvisFauna <{sender}>"
    msg["To"]      = dest_email
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, dest_email, msg.as_string())
        return True, ""
    except smtplib.SMTPAuthenticationError:
        return False, "auth_error"
    except Exception as e:
        return False, str(e)


# ═══════════════════════════════════════════════════════════════════════════════
# ─── CONFIGURACIÓN DE MODELOS (GOOGLE DRIVE)
# ═══════════════════════════════════════════════════════════════════════════════

GDRIVE_MODEL_1_ID = '1FcU2jJEYqU0c971n9WIytWfFaTD018z-'   # inception_finetuned.h5
GDRIVE_MODEL_2_ID = '1ZvA60kdraCRDw8wGx83FqfUkwWOeykPC'   # VGG16_finetuned.h5
MODEL_1_PATH      = 'inception_finetuned.h5'
MODEL_2_PATH      = 'VGG16_finetuned.h5'
IMG_SIZE          = (224, 224)

CLASSES_MODEL_1 = [
    'Anhinga anhinga', 'Butorides striata', 'Chamaepetes goudotii',
    'Colinus cristatus', 'Nycticorax nycticorax', 'Ortalis guttata',
    'Penelope montagnii', 'Podilymbus podiceps', 'Tachybaptus dominicus',
    'Tigrisoma lineatum'
]
CLASSES_MODEL_2 = [
    'Anhinga anhinga', 'Butorides striata', 'Chamaepetes goudotii',
    'Colinus cristatus', 'Nycticorax nycticorax', 'Ortalis guttata',
    'Penelope montagnii', 'Podilymbus podiceps', 'Tachybaptus dominicus',
    'Tigrisoma lineatum'
]
COMMON_NAMES = {
    'Anhinga anhinga':       'Pato aguja',
    'Butorides striata':     'Garza verdosa',
    'Chamaepetes goudotii':  'Pava caucana',
    'Colinus cristatus':     'Codorniz crestada',
    'Nycticorax nycticorax': 'Garza nocturna',
    'Ortalis guttata':       'Guacharaca moteada',
    'Penelope montagnii':    'Pava andina',
    'Podilymbus podiceps':   'Zambullidor piquipinto',
    'Tachybaptus dominicus': 'Zambullidor menor',
    'Tigrisoma lineatum':    'Garza tigre rayada'
}


# ═══════════════════════════════════════════════════════════════════════════════
# ─── DESCARGA Y CARGA DE MODELOS DESDE GOOGLE DRIVE
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def download_model_from_drive(file_id, output_path):
    """Descarga un modelo desde Google Drive si no existe localmente."""
    if not os.path.exists(output_path):
        try:
            with st.spinner(f'Descargando {output_path} desde Google Drive...'):
                url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(url, output_path, quiet=False)
            st.success(f'✅ {output_path} descargado exitosamente')
        except Exception as e:
            st.error(f'Error al descargar {output_path}: {e}')
            return False
    return True

@st.cache_resource
def load_models():
    """Descarga y carga ambos modelos pre-entrenados desde Google Drive."""
    try:
        if not download_model_from_drive(GDRIVE_MODEL_1_ID, MODEL_1_PATH):
            return None
        if not download_model_from_drive(GDRIVE_MODEL_2_ID, MODEL_2_PATH):
            return None

        model1 = tf.keras.models.load_model(MODEL_1_PATH, compile=False)
        model2 = tf.keras.models.load_model(MODEL_2_PATH, compile=False)

        model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return {"Modelo 1 (Inception)": model1, "Modelo 2 (VGG16)": model2}
    except Exception as e:
        st.error(f"Error al cargar los modelos: {e}")
        st.info("Verifica que:")
        st.write("- Los IDs de Google Drive sean correctos")
        st.write("- Los archivos estén compartidos como 'Cualquiera con el enlace'")
        st.write(f"- ID Modelo 1: {GDRIVE_MODEL_1_ID}")
        st.write(f"- ID Modelo 2: {GDRIVE_MODEL_2_ID}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# ─── PREDICCIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def predict(image, model_key, models_dict):
    """Prepara la imagen y realiza la predicción."""
    model   = models_dict[model_key]
    classes = CLASSES_MODEL_1 if "Modelo 1" in model_key else CLASSES_MODEL_2

    img       = image.convert('RGB').resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions          = model.predict(img_array)
    predicted_class_index = int(np.argmax(predictions, axis=1)[0])
    predicted_class      = classes[predicted_class_index]
    confidence           = float(predictions[0][predicted_class_index]) * 100

    return predicted_class, confidence, predictions[0]


# ═══════════════════════════════════════════════════════════════════════════════
# ─── PANTALLA DE AUTENTICACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def auth_screen():
    # Ocultar el botón colapsar sidebar en la pantalla de auth
    st.markdown("""
    <style>
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] { display: none !important; }
    section[data-testid="stSidebar"] { display: none !important; }
    /* ── Tabs auth mejorados ── */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 0 !important;
        padding: 0 !important;
        gap: 0 !important;
        margin-bottom: 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        color: rgba(255,255,255,0.42) !important;
        padding: 14px 28px !important;
        border-radius: 0 !important;
        border-bottom: 2px solid transparent !important;
        letter-spacing: 0.01em;
        transition: color 0.2s, border-color 0.2s !important;
        background: transparent !important;
        margin-right: 4px !important;
    }
    .stTabs [aria-selected="true"] {
        color: #60a5fa !important;
        background: transparent !important;
        border-bottom: 2px solid #2563eb !important;
        border-radius: 0 !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: rgba(255,255,255,0.75) !important;
        background: rgba(255,255,255,0.04) !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 2rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="auth-container animate-in">', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; margin-bottom:2.5rem; padding-top:2rem;">
        <span class="bird-icon">🦅</span>
        <div class="auth-logo">AvisFauna</div>
        <div class="auth-tagline">Identificación inteligente de aves colombianas</div>
    </div>
    """, unsafe_allow_html=True)

    tab_login, tab_register, tab_reset = st.tabs(["  Iniciar sesión  ", "  Registrarse  ", "  Olvidé mi contraseña  "])

    # ── Login ──────────────────────────────────────────────────────────────
    with tab_login:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.form("form_login"):
            email    = st.text_input("Correo electrónico", placeholder="usuario@correo.com")
            password = st.text_input("Contraseña", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Entrar →", use_container_width=True)

        if submitted:
            if not email or not password:
                st.error("Por favor completa todos los campos.")
            else:
                ensure_admin_role()
                users = load_users()
                h     = hash_password(password)
                if email in users and users[email]["password"] == h:
                    role = get_user_role(email)
                    st.session_state.authenticated = True
                    st.session_state.user_email    = email
                    st.session_state.user_name     = users[email].get("name", email.split("@")[0])
                    st.session_state.user_role     = role
                    st.session_state.current_page  = "clasificador"
                    users[email]["last_login"] = datetime.now().isoformat()
                    save_users(users)
                    # Persistir sesión en disco para sobrevivir recargas
                    st.session_state.login_time = datetime.now().isoformat()
                    save_session(email, st.session_state.user_name, role)
                    st.success(f"¡Bienvenido/a, {st.session_state.user_name}!")
                    st.rerun()
                else:
                    st.error("Correo o contraseña incorrectos.")

    # ── Registro ───────────────────────────────────────────────────────────
    with tab_register:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.form("form_register"):
            name      = st.text_input("Nombre completo", placeholder="Ana García")
            reg_email = st.text_input("Correo electrónico", placeholder="usuario@correo.com")
            reg_pass  = st.text_input("Contraseña", type="password", placeholder="Mínimo 8 caracteres")
            reg_pass2 = st.text_input("Confirmar contraseña", type="password", placeholder="Repite tu contraseña")
            submitted_r = st.form_submit_button("Crear cuenta →", use_container_width=True)

        if submitted_r:
            users = load_users()
            ok    = True
            if not name or not reg_email or not reg_pass or not reg_pass2:
                st.error("Completa todos los campos."); ok = False
            elif not validate_email(reg_email):
                st.error("El correo no tiene un formato válido."); ok = False
            elif reg_email in users:
                st.error("Este correo ya está registrado."); ok = False
            elif len(reg_pass) < 8:
                st.error("La contraseña debe tener al menos 8 caracteres."); ok = False
            elif reg_pass != reg_pass2:
                st.error("Las contraseñas no coinciden."); ok = False

            if ok:
                role = "administrador" if reg_email == ADMIN_EMAIL else "usuario"
                users[reg_email] = {
                    "name":       name,
                    "password":   hash_password(reg_pass),
                    "role":       role,
                    "created_at": datetime.now().isoformat(),
                    "last_login": None,
                }
                save_users(users)
                st.success("✅ Cuenta creada. ¡Ya puedes iniciar sesión!")

    # ── Olvidé mi contraseña ───────────────────────────────────────────────
    with tab_reset:
        st.markdown("<br>", unsafe_allow_html=True)

        if "reset_step" not in st.session_state:
            st.session_state.reset_step = 1

        # ── PASO 1: solicitar código ───────────────────────────────────────
        if st.session_state.reset_step == 1:
            st.markdown("""
            <div style="text-align:center; margin-bottom:1.4rem;">
                <div style="font-size:2rem; margin-bottom:6px;">🔐</div>
                <div class="display-subtitle">
                    Ingresa tu correo registrado y te enviaremos<br>un código de verificación de 8 caracteres.
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.form("form_reset_req"):
                reset_email = st.text_input("Correo electrónico", placeholder="usuario@correo.com")
                sub_req     = st.form_submit_button("📨 Enviar código de recuperación", use_container_width=True)

            if sub_req:
                users = load_users()
                if not reset_email:
                    st.error("Ingresa tu correo.")
                elif not validate_email(reset_email):
                    st.error("El formato del correo no es válido.")
                elif reset_email not in users:
                    st.error("No encontramos una cuenta con ese correo.")
                else:
                    token  = generate_token(8).upper()
                    tokens = load_reset_tokens()
                    tokens[reset_email] = {
                        "token":   token,
                        "expires": (datetime.now() + timedelta(minutes=15)).isoformat()
                    }
                    save_reset_tokens(tokens)
                    st.session_state.reset_email_target = reset_email

                    dest_name   = users[reset_email].get("name", reset_email.split("@")[0])
                    sent, error = send_reset_email(reset_email, dest_name, token)

                    if sent:
                        st.session_state.reset_step       = 2
                        st.session_state.reset_email_sent = True
                        st.rerun()
                    elif error == "no_config":
                        st.session_state.reset_step       = 2
                        st.session_state.reset_email_sent = False
                        st.session_state.reset_demo_token = token
                        st.rerun()
                    elif error == "auth_error":
                        st.error("Error de autenticacion SMTP. Verifica las credenciales en secrets.toml.")
                    else:
                        st.error(f"No se pudo enviar el correo: {error}")

        # ── PASO 2: ingresar código + nueva contraseña ─────────────────────
        elif st.session_state.reset_step == 2:
            target_email = st.session_state.get("reset_email_target", "")
            email_sent   = st.session_state.get("reset_email_sent", False)
            demo_token   = st.session_state.get("reset_demo_token", "")

            if email_sent:
                st.markdown(f"""
                <div style="background:rgba(134,239,172,0.08); border:1px solid rgba(134,239,172,0.25);
                            border-radius:12px; padding:1.1rem 1.4rem; margin-bottom:1.4rem; text-align:center;">
                    <div style="font-size:1.5rem; margin-bottom:6px;">📬</div>
                    <div style="font-size:0.92rem; color:#86efac; font-weight:500;">
                        Código enviado exitosamente
                    </div>
                    <div style="font-size:0.8rem; color:rgba(255,255,255,0.5); margin-top:4px;">
                        Revisa la bandeja de entrada de <strong style="color:#93c5fd;">{target_email}</strong><br>
                        (también revisa spam / correo no deseado)
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:rgba(234,179,8,0.08); border:1px solid rgba(234,179,8,0.25);
                            border-radius:12px; padding:1.1rem 1.4rem; margin-bottom:1rem; text-align:center;">
                    <div style="font-size:1.3rem; margin-bottom:6px;">⚙️</div>
                    <div style="font-size:0.85rem; color:#fde68a; font-weight:500;">Modo demo — SMTP no configurado</div>
                    <div style="font-size:0.78rem; color:rgba(255,255,255,0.45); margin-top:3px;">
                        Configura <code style="color:#fcd34d;">.streamlit/secrets.toml</code> para envio real.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if demo_token:
                    st.markdown(f"""
                    <div style="background:rgba(37,99,235,0.12); border:2px solid rgba(96,165,250,0.30);
                                border-radius:12px; padding:1.2rem; text-align:center; margin-bottom:1.2rem;">
                        <div style="font-size:0.73rem; color:rgba(255,255,255,0.42); text-transform:uppercase;
                                    letter-spacing:0.1em; margin-bottom:8px;">Tu código de recuperación</div>
                        <div style="font-size:2rem; font-weight:700; color:#60a5fa;
                                    letter-spacing:0.25em; font-family:monospace;">
                            {demo_token}
                        </div>
                        <div style="font-size:0.75rem; color:rgba(255,255,255,0.38); margin-top:8px;">
                            Valido durante 15 minutos
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with st.form("form_reset_verify"):
                input_token = st.text_input(
                    "Código de verificación",
                    placeholder="Ej: A3BX7K2M",
                    help="Código de 8 caracteres recibido por correo"
                )
                new_pass  = st.text_input("Nueva contraseña",     type="password", placeholder="Mínimo 8 caracteres")
                new_pass2 = st.text_input("Confirmar contraseña", type="password", placeholder="Repite tu nueva contraseña")
                sub_verify = st.form_submit_button("🔒 Restablecer contraseña", use_container_width=True)

            col_b, _ = st.columns([1, 3])
            with col_b:
                if st.button("← Volver", use_container_width=True):
                    for k in ["reset_step", "reset_email_target", "reset_email_sent", "reset_demo_token"]:
                        st.session_state.pop(k, None)
                    st.rerun()

            if sub_verify:
                tokens = load_reset_tokens()
                ok     = True

                if not input_token or not new_pass or not new_pass2:
                    st.error("Completa todos los campos."); ok = False
                elif target_email not in tokens:
                    st.error("No hay un código activo. Solicita uno nuevo."); ok = False
                else:
                    t_data  = tokens[target_email]
                    expired = datetime.now() > datetime.fromisoformat(t_data["expires"])
                    if expired:
                        st.error("El código ha expirado. Vuelve y solicita uno nuevo."); ok = False
                    elif input_token.strip().upper() != t_data["token"]:
                        st.error("Código incorrecto. Verifica e intenta de nuevo."); ok = False

                if ok:
                    if len(new_pass) < 8:
                        st.error("La contraseña debe tener al menos 8 caracteres.")
                    elif new_pass != new_pass2:
                        st.error("Las contraseñas no coinciden.")
                    else:
                        users = load_users()
                        users[target_email]["password"] = hash_password(new_pass)
                        save_users(users)
                        del tokens[target_email]
                        save_reset_tokens(tokens)
                        for k in ["reset_step", "reset_email_target", "reset_email_sent", "reset_demo_token"]:
                            st.session_state.pop(k, None)
                        st.success("Contraseña restablecida correctamente. Ya puedes iniciar sesion!")
                        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ─── HELPERS SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def format_connected_time():
    """Calcula cuánto tiempo lleva el usuario conectado en esta sesión."""
    login_str = st.session_state.get("login_time")
    if not login_str:
        return "—"
    try:
        delta = datetime.now() - datetime.fromisoformat(login_str)
        total = int(delta.total_seconds())
        h, rem = divmod(total, 3600)
        m, s   = divmod(rem, 60)
        if h > 0:
            return f"{h}h {m:02d}m"
        elif m > 0:
            return f"{m}m {s:02d}s"
        else:
            return f"{s}s"
    except Exception:
        return "—"

def get_user_top_species(email, limit=5):
    """Devuelve las N especies más consultadas por el usuario."""
    stats = load_stats()
    my_preds = [p for p in stats["predictions"] if p["user_email"] == email]
    counts = Counter(p["species"] for p in my_preds)
    total  = len(my_preds)
    return counts.most_common(limit), total


# ═══════════════════════════════════════════════════════════════════════════════
# ─── SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar(models_dict):
    role      = st.session_state.get("user_role", "usuario")
    av_color  = avatar_color(role)
    current   = st.session_state.get("current_page", "clasificador")
    email     = st.session_state.get("user_email", "")
    name      = st.session_state.get("user_name", "")

    with st.sidebar:

        # ── Logo ──────────────────────────────────────────────────────────
        st.markdown("""
        <div style="padding:1rem 0 1rem 0;">
            <div style="font-family:'Playfair Display',serif; font-size:1.4rem; font-weight:700; color:#fff;">
                🦅 AvisFauna
            </div>
            <div style="font-size:0.74rem; color:rgba(255,255,255,0.32); margin-top:2px;">
                Identificación inteligente
            </div>
        </div>
        <div style="height:1px; background:rgba(255,255,255,0.08); margin-bottom:1rem;"></div>
        """, unsafe_allow_html=True)

        # ── Tarjeta de usuario ────────────────────────────────────────────
        connected_time = format_connected_time()
        role_info      = ROLES.get(role, ROLES["usuario"])

        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.10);
                    border-radius:14px; padding:1rem; margin-bottom:1rem;">

            <!-- Avatar + nombre -->
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:0.9rem;">
                <div style="width:42px;height:42px;border-radius:50%;background:{av_color};
                            display:flex;align-items:center;justify-content:center;
                            font-weight:700;font-size:1.05rem;color:#fff;flex-shrink:0;
                            box-shadow:0 0 0 3px rgba(255,255,255,0.08);">
                    {name[0].upper() if name else "?"}
                </div>
                <div style="min-width:0;">
                    <div style="color:#fff;font-size:0.9rem;font-weight:600;
                                white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                        {name}
                    </div>
                    <div style="font-size:0.72rem;color:rgba(255,255,255,0.38);
                                white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                        {email}
                    </div>
                </div>
            </div>

            <!-- Rol -->
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding:6px 0; border-top:1px solid rgba(255,255,255,0.07);">
                <span style="font-size:0.72rem; color:rgba(255,255,255,0.38);
                             text-transform:uppercase; letter-spacing:0.07em;">Rol</span>
                {role_badge_html(role)}
            </div>

            <!-- Tiempo conectado -->
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding:6px 0; border-top:1px solid rgba(255,255,255,0.07);">
                <span style="font-size:0.72rem; color:rgba(255,255,255,0.38);
                             text-transform:uppercase; letter-spacing:0.07em;">Conectado</span>
                <span style="font-size:0.82rem; font-weight:600; color:#60a5fa;">
                    ⏱ {connected_time}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Navegación ────────────────────────────────────────────────────
        st.markdown('<div class="label-muted" style="margin-bottom:0.5rem;">Navegación</div>', unsafe_allow_html=True)

        nav_items = [("clasificador", "🔬", "Clasificador")]
        if can(role, "ver_estadisticas"):
            nav_items.append(("estadisticas", "📊", "Mis estadísticas"))
        if can(role, "dashboard_admin"):
            nav_items.append(("dashboard", "🛡️", "Panel de admin"))
            nav_items.append(("roles", "👥", "Gestión de roles"))

        for page_id, icon, label in nav_items:
            is_active = (current == page_id)
            bg     = "rgba(37,99,235,0.28)" if is_active else "rgba(255,255,255,0.03)"
            color  = "#93c5fd"              if is_active else "rgba(255,255,255,0.62)"
            border = "1px solid rgba(96,165,250,0.25)" if is_active else "1px solid rgba(255,255,255,0.06)"
            st.markdown(f"""
            <div style="background:{bg};border:{border};border-radius:8px;
                        padding:8px 12px;margin-bottom:4px;color:{color};font-size:0.88rem;">
                {icon} {label}
            </div>
            """, unsafe_allow_html=True)
            if st.button(label, key=f"nav_{page_id}", use_container_width=True):
                st.session_state.current_page = page_id
                st.rerun()

        # ── Modelo ────────────────────────────────────────────────────────
        st.markdown("""
        <div style="height:1px; background:rgba(255,255,255,0.08); margin:0.8rem 0 0.6rem 0;"></div>
        <div class="label-muted" style="margin-bottom:0.4rem;">Modelo de clasificación</div>
        """, unsafe_allow_html=True)

        model_choice = st.selectbox(
            label="Modelo",
            options=list(models_dict.keys()),
            label_visibility="collapsed"
        )

        # ── Mini estadísticas de especies (sidebar) ───────────────────────
        st.markdown("""
        <div style="height:1px; background:rgba(255,255,255,0.08); margin:0.8rem 0 0.7rem 0;"></div>
        <div class="label-muted" style="margin-bottom:0.6rem;">Mis aves más examinadas</div>
        """, unsafe_allow_html=True)

        top_species, total_user = get_user_top_species(email, limit=5)

        if total_user == 0:
            st.markdown("""
            <div style="font-size:0.78rem; color:rgba(255,255,255,0.3); text-align:center;
                        padding:0.8rem 0; font-style:italic;">
                Aún no hay clasificaciones
            </div>
            """, unsafe_allow_html=True)
        else:
            max_cnt = top_species[0][1] if top_species else 1
            for sp, cnt in top_species:
                common  = COMMON_NAMES.get(sp, "")
                pct_bar = cnt / max_cnt * 100
                pct_tot = cnt / total_user * 100
                short_sp = sp.split(" ")[1] if " " in sp else sp  # solo epíteto
                st.markdown(f"""
                <div style="margin-bottom:9px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                        <div style="min-width:0; flex:1;">
                            <div style="font-size:0.77rem; font-style:italic;
                                        color:rgba(255,255,255,0.72);
                                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                                {sp}
                            </div>
                            <div style="font-size:0.68rem; color:rgba(255,255,255,0.35);">
                                {common}
                            </div>
                        </div>
                        <div style="text-align:right; flex-shrink:0; margin-left:6px;">
                            <div style="font-size:0.8rem; font-weight:700; color:#93c5fd;">{cnt}×</div>
                            <div style="font-size:0.67rem; color:rgba(255,255,255,0.3);">{pct_tot:.0f}%</div>
                        </div>
                    </div>
                    <div style="background:rgba(255,255,255,0.06); border-radius:3px; height:4px;">
                        <div style="width:{pct_bar:.1f}%; height:4px; border-radius:3px;
                                    background:linear-gradient(90deg,#1d4ed8,#60a5fa);"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="font-size:0.71rem; color:rgba(255,255,255,0.3);
                        text-align:right; margin-top:2px;">
                {total_user} clasificaciones totales
            </div>
            """, unsafe_allow_html=True)

        # ── Cerrar sesión ─────────────────────────────────────────────────
        st.markdown("""
        <div style="height:1px; background:rgba(255,255,255,0.08); margin:0.9rem 0 0.7rem 0;"></div>
        """, unsafe_allow_html=True)

        if st.button("Cerrar sesión", use_container_width=True):
            clear_session()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    return model_choice


# ═══════════════════════════════════════════════════════════════════════════════
# ─── PÁGINA: CLASIFICADOR
# ═══════════════════════════════════════════════════════════════════════════════

def page_clasificador(models_dict, model_choice):
    col_h1, col_h2 = st.columns([2, 1])
    with col_h1:
        st.markdown(f"""
        <div class="animate-in">
            <div class="display-title">Identificador<br>de Aves 🦅</div>
            <div class="display-subtitle" style="margin-top:0.7rem; max-width:500px;">
                Sube una fotografía y nuestro modelo de visión artificial identificará
                la especie con alta precisión.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_h2:
        st.markdown(f"""
        <div style="text-align:right; padding-top:1rem;">
            <div class="model-chip"><span>🧠</span> {model_choice}</div>
            <div style="font-size:0.8rem; color:rgba(255,255,255,0.35); margin-top:8px;">
                {len(CLASSES_MODEL_1)} especies · Colombia
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_upload, col_result = st.columns([1, 1], gap="large")

    # ── Columna izquierda: upload ──────────────────────────────────────────
    with col_upload:
        st.markdown('<div class="card animate-in">', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="margin-bottom:1.2rem;">📤 Sube tu imagen</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            label="Elige un archivo",
            type=["jpg", "png", "jpeg"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen cargada", use_column_width=True, output_format="PNG")
            st.markdown(f"""
            <div style="display:flex; gap:10px; margin-top:0.8rem; flex-wrap:wrap;">
                <span style="font-size:0.78rem; color:rgba(255,255,255,0.4);">{image.width} × {image.height} px</span>
                <span style="font-size:0.78rem; color:rgba(255,255,255,0.4);">·</span>
                <span style="font-size:0.78rem; color:rgba(255,255,255,0.4);">{uploaded_file.name}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center; padding:2.5rem 1rem; color:rgba(255,255,255,0.3);">
                <div style="font-size:2.5rem; margin-bottom:0.8rem;">🖼️</div>
                <div style="font-size:0.9rem;">Arrastra o selecciona una imagen JPG / PNG</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Columna derecha: resultado ─────────────────────────────────────────
    with col_result:
        st.markdown('<div class="card animate-in" style="min-height:320px;">', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="margin-bottom:1.2rem;">🔬 Resultado</div>', unsafe_allow_html=True)

        if uploaded_file:
            if st.button("✦ Identificar especie", use_container_width=True):
                with st.spinner("Analizando imagen..."):
                    try:
                        species, confidence, all_preds = predict(image, model_choice, models_dict)
                        common         = COMMON_NAMES.get(species, "—")
                        active_classes = CLASSES_MODEL_1 if "Modelo 1" in model_choice else CLASSES_MODEL_2

                        st.session_state.last_result = {
                            "species":    species,
                            "common":     common,
                            "confidence": confidence,
                            "all_preds":  all_preds.tolist(),
                            "classes":    active_classes
                        }
                        record_prediction(
                            st.session_state.user_email,
                            species, confidence, model_choice
                        )
                    except Exception as e:
                        st.error(f"Error durante la predicción: {e}")

        if "last_result" in st.session_state and uploaded_file:
            r = st.session_state.last_result
            st.markdown(f"""
            <div style="margin-bottom:1.5rem;">
                <div class="result-label">Especie identificada</div>
                <div class="result-species">{r['species']}</div>
                <div style="font-size:0.9rem; color:rgba(255,255,255,0.5); margin-top:2px;">{r['common']}</div>
            </div>
            <div style="margin-bottom:1.5rem;">
                <div class="result-label">Confianza</div>
                <div class="result-confidence">{r['confidence']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(r['confidence'] / 100)
        else:
            st.markdown("""
            <div style="text-align:center; padding:3rem 1rem; color:rgba(255,255,255,0.25);">
                <div style="font-size:2rem; margin-bottom:0.5rem;">🔍</div>
                <div style="font-size:0.9rem;">Sube una imagen y pulsa "Identificar especie"</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Distribución de probabilidades ──────────────────────────────────────
    if "last_result" in st.session_state and uploaded_file:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card animate-in">', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="margin-bottom:1.4rem;">📊 Distribución de probabilidades</div>', unsafe_allow_html=True)

        r             = st.session_state.last_result
        result_classes = r.get("classes", CLASSES_MODEL_1)
        sorted_pairs  = sorted(zip(result_classes, r["all_preds"]), key=lambda x: x[1], reverse=True)

        cols = st.columns(2)
        for i, (cls, prob) in enumerate(sorted_pairs):
            pct    = prob * 100
            common = COMMON_NAMES.get(cls, "")
            is_top = (cls == r["species"])
            color  = "#60a5fa" if is_top else "#475569"

            with cols[i % 2]:
                st.markdown(f"""
                <div style="margin-bottom:14px;">
                    <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:5px;">
                        <div>
                            <span style="font-size:0.82rem; font-style:italic; color:{'#60a5fa' if is_top else 'rgba(255,255,255,0.65)'};">{cls}</span>
                            <span style="font-size:0.7rem; color:rgba(255,255,255,0.35); margin-left:6px;">{common}</span>
                        </div>
                        <span style="font-size:0.82rem; font-weight:600; color:{'#93c5fd' if is_top else 'rgba(255,255,255,0.5)'};">{pct:.1f}%</span>
                    </div>
                    <div class="prob-bar-container">
                        <div class="prob-bar-fill" style="width:{pct:.1f}%; background:{color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ─── PÁGINA: MIS ESTADÍSTICAS (experto + admin)
# ═══════════════════════════════════════════════════════════════════════════════

def page_estadisticas():
    email    = st.session_state.user_email
    stats    = load_stats()
    my_preds = [p for p in stats["predictions"] if p["user_email"] == email]

    st.markdown('<div class="page-title animate-in">📊 Mis estadísticas</div>', unsafe_allow_html=True)
    st.markdown('<div class="display-subtitle" style="margin-bottom:2rem;">Tu actividad de clasificación en AvisFauna</div>', unsafe_allow_html=True)

    total          = len(my_preds)
    avg_conf       = round(sum(p["confidence"] for p in my_preds) / total, 1) if total else 0
    species_counts = Counter(p["species"] for p in my_preds)
    top_species    = species_counts.most_common(1)[0][0] if species_counts else "—"
    top_common     = COMMON_NAMES.get(top_species, top_species)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    for col, (icon, val, label) in zip([c1, c2, c3, c4], [
        ("🔬", str(total),               "Clasificaciones"),
        ("🎯", f"{avg_conf}%",           "Confianza media"),
        ("🦅", top_common[:16],          "Ave más vista"),
        ("🧠", str(len(set(p["model"] for p in my_preds))), "Modelos usados"),
    ]):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-icon">{icon}</div>
                <div class="stat-value">{val}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if total == 0:
        st.markdown('<div class="alert-info">Aún no has realizado ninguna clasificación. ¡Ve al clasificador y sube tu primera imagen!</div>', unsafe_allow_html=True)
        return

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="margin-bottom:1.2rem;">🦅 Especies clasificadas</div>', unsafe_allow_html=True)
        for sp, cnt in species_counts.most_common():
            pct    = cnt / total * 100
            common = COMMON_NAMES.get(sp, "")
            st.markdown(f"""
            <div style="margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <div>
                        <span style="font-size:0.83rem; font-style:italic; color:rgba(255,255,255,0.75);">{sp}</span>
                        <span style="font-size:0.72rem; color:rgba(255,255,255,0.35); margin-left:6px;">{common}</span>
                    </div>
                    <span style="font-size:0.83rem; color:#93c5fd; font-weight:600;">{cnt}×</span>
                </div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill" style="width:{pct:.1f}%; background:#2563eb;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="margin-bottom:1.2rem;">🕘 Actividad reciente</div>', unsafe_allow_html=True)
        recent = sorted(my_preds, key=lambda x: x["timestamp"], reverse=True)[:8]
        for p in recent:
            ts     = datetime.fromisoformat(p["timestamp"]).strftime("%d %b %Y · %H:%M")
            common = COMMON_NAMES.get(p["species"], p["species"])
            conf_c = "#86efac" if p["confidence"] >= 80 else "#fde68a" if p["confidence"] >= 60 else "#fca5a5"
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.06);">
                <div>
                    <div style="font-size:0.83rem; color:rgba(255,255,255,0.8); font-style:italic;">{p['species']}</div>
                    <div style="font-size:0.73rem; color:rgba(255,255,255,0.35);">{ts}</div>
                </div>
                <span style="font-size:0.82rem; font-weight:600; color:{conf_c};">{p['confidence']:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ─── PÁGINA: PANEL DE ADMINISTRACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def page_dashboard_admin():
    users     = load_users()
    stats     = load_stats()
    all_preds = stats["predictions"]

    st.markdown('<div class="page-title animate-in">🛡️ Panel de administración</div>', unsafe_allow_html=True)
    st.markdown('<div class="display-subtitle" style="margin-bottom:2rem;">Gestión de usuarios, roles y estadísticas globales</div>', unsafe_allow_html=True)

    # KPIs globales
    total_users   = len(users)
    roles_count   = Counter(u.get("role", "usuario") for u in users.values())
    total_preds   = len(all_preds)
    avg_conf_g    = round(sum(p["confidence"] for p in all_preds) / total_preds, 1) if total_preds else 0
    active_today  = len(set(
        p["user_email"] for p in all_preds
        if datetime.fromisoformat(p["timestamp"]).date() == datetime.now().date()
    ))

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, (icon, val, label) in zip([c1,c2,c3,c4,c5], [
        ("👥", str(total_users),                              "Usuarios totales"),
        ("👑", str(roles_count.get("administrador", 0)),      "Administradores"),
        ("🔬", str(roles_count.get("experto", 0)),            "Expertos"),
        ("🖼️", str(total_preds),                             "Clasificaciones"),
        ("⚡", str(active_today),                             "Activos hoy"),
    ]):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-icon">{icon}</div>
                <div class="stat-value">{val}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab_users, tab_stats, tab_species = st.tabs(["👥 Gestión de usuarios", "📈 Estadísticas globales", "🦅 Por especie"])

    # ── TAB 1: Gestión de usuarios ─────────────────────────────────────────
    with tab_users:
        st.markdown("<br>", unsafe_allow_html=True)
        search = st.text_input("🔍 Buscar por nombre o correo", placeholder="Escribe para filtrar...")
        st.markdown("<br>", unsafe_allow_html=True)

        for email, data in users.items():
            name       = data.get("name", "Sin nombre")
            role       = data.get("role", "usuario")
            created    = data.get("created_at", "")[:10] if data.get("created_at") else "—"
            last_login = data.get("last_login", "")[:10] if data.get("last_login") else "Nunca"
            user_preds = len([p for p in all_preds if p["user_email"] == email])

            if search and search.lower() not in name.lower() and search.lower() not in email.lower():
                continue

            av_col = avatar_color(role)
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.09);
                        border-radius:12px; padding:1rem 1.2rem; margin-bottom:10px;">
                <div style="display:flex; align-items:center; gap:14px; flex-wrap:wrap;">
                    <div style="width:40px;height:40px;border-radius:50%;background:{av_col};
                                display:flex;align-items:center;justify-content:center;
                                font-weight:700;font-size:1rem;color:#fff;flex-shrink:0;">
                        {name[0].upper()}
                    </div>
                    <div style="flex:1; min-width:160px;">
                        <div style="font-size:0.92rem;color:#fff;font-weight:500;">{name}</div>
                        <div style="font-size:0.75rem;color:rgba(255,255,255,0.4);">{email}</div>
                    </div>
                    <div style="display:flex;gap:20px;flex-wrap:wrap;align-items:center;">
                        <div style="text-align:center;">
                            <div class="label-muted">Rol</div>
                            <div style="margin-top:3px;">{role_badge_html(role)}</div>
                        </div>
                        <div style="text-align:center;">
                            <div class="label-muted">Registro</div>
                            <div style="font-size:0.8rem;color:rgba(255,255,255,0.65);margin-top:2px;">{created}</div>
                        </div>
                        <div style="text-align:center;">
                            <div class="label-muted">Último acceso</div>
                            <div style="font-size:0.8rem;color:rgba(255,255,255,0.65);margin-top:2px;">{last_login}</div>
                        </div>
                        <div style="text-align:center;">
                            <div class="label-muted">Clasificaciones</div>
                            <div style="font-size:1.1rem;color:#93c5fd;font-weight:700;margin-top:2px;">{user_preds}</div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Admin principal → rol fijo
            if email == ADMIN_EMAIL:
                st.markdown('<div class="alert-warn" style="margin-bottom:8px;">👑 Cuenta de administrador principal — rol permanente.</div>', unsafe_allow_html=True)
                continue

            col_sel, col_btn, _ = st.columns([2, 1, 3])
            with col_sel:
                new_role = st.selectbox(
                    "Cambiar rol",
                    options=list(ROLES.keys()),
                    index=list(ROLES.keys()).index(role),
                    key=f"role_sel_{email}",
                    label_visibility="collapsed"
                )
            with col_btn:
                if st.button("Aplicar", key=f"role_btn_{email}", use_container_width=True):
                    users_fresh = load_users()
                    users_fresh[email]["role"] = new_role
                    save_users(users_fresh)
                    st.success(f"Rol de **{name}** actualizado a **{ROLES[new_role]['label']}**.")
                    st.rerun()

            st.markdown("<div style='margin-bottom:4px;'></div>", unsafe_allow_html=True)

    # ── TAB 2: Estadísticas globales ───────────────────────────────────────
    with tab_stats:
        st.markdown("<br>", unsafe_allow_html=True)

        if not all_preds:
            st.markdown('<div class="alert-info">Aún no hay clasificaciones registradas en el sistema.</div>', unsafe_allow_html=True)
        else:
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title" style="margin-bottom:1.2rem;">👤 Clasificaciones por usuario</div>', unsafe_allow_html=True)
                user_pred_count = Counter(p["user_email"] for p in all_preds)
                max_c = max(user_pred_count.values())
                for em, cnt in user_pred_count.most_common(10):
                    nm  = users.get(em, {}).get("name", em.split("@")[0])
                    rl  = users.get(em, {}).get("role", "usuario")
                    pct = cnt / max_c * 100
                    st.markdown(f"""
                    <div style="margin-bottom:11px;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px;flex-wrap:wrap;gap:4px;">
                            <span style="font-size:0.82rem;color:rgba(255,255,255,0.75);">{nm} {role_badge_html(rl)}</span>
                            <span style="font-size:0.82rem;color:#93c5fd;font-weight:600;">{cnt}</span>
                        </div>
                        <div class="prob-bar-container">
                            <div class="prob-bar-fill" style="width:{pct:.1f}%;background:#2563eb;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_b:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title" style="margin-bottom:1.2rem;">🧠 Uso por modelo</div>', unsafe_allow_html=True)
                model_count = Counter(p["model"] for p in all_preds)
                for model, cnt in model_count.most_common():
                    pct = cnt / total_preds * 100
                    st.markdown(f"""
                    <div style="margin-bottom:12px;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                            <span style="font-size:0.83rem;color:rgba(255,255,255,0.75);">{model}</span>
                            <span style="font-size:0.83rem;color:#93c5fd;font-weight:600;">{cnt} · {pct:.0f}%</span>
                        </div>
                        <div class="prob-bar-container">
                            <div class="prob-bar-fill" style="width:{pct:.1f}%;background:#7c3aed;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-title" style="margin-bottom:0.8rem;">🎯 Confianza promedio global</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div style="text-align:center; padding:1rem 0;">
                    <div style="font-size:3rem;font-weight:700;color:#60a5fa;">{avg_conf_g}%</div>
                    <div class="stat-label">sobre {total_preds} clasificaciones</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Tabla: últimas clasificaciones del sistema
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title" style="margin-bottom:1.2rem;">🕘 Últimas clasificaciones del sistema</div>', unsafe_allow_html=True)

            recent_all = sorted(all_preds, key=lambda x: x["timestamp"], reverse=True)[:12]
            header_cols = st.columns([2, 2, 1, 1, 2])
            for hc, lbl in zip(header_cols, ["Usuario", "Especie", "Confianza", "Modelo", "Fecha"]):
                with hc:
                    st.markdown(f'<div class="label-muted">{lbl}</div>', unsafe_allow_html=True)

            st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
            for p in recent_all:
                nm     = users.get(p["user_email"], {}).get("name", p["user_email"].split("@")[0])
                ts     = datetime.fromisoformat(p["timestamp"]).strftime("%d %b · %H:%M")
                conf_c = "#86efac" if p["confidence"] >= 80 else "#fde68a" if p["confidence"] >= 60 else "#fca5a5"
                mod_s  = "Inception" if "Modelo 1" in p["model"] else "VGG16"
                row_cols = st.columns([2, 2, 1, 1, 2])
                for rc, html in zip(row_cols, [
                    f'<span style="font-size:0.82rem;color:rgba(255,255,255,0.75);">{nm}</span>',
                    f'<span style="font-size:0.82rem;font-style:italic;color:rgba(255,255,255,0.65);">{p["species"]}</span>',
                    f'<span style="font-size:0.82rem;font-weight:600;color:{conf_c};">{p["confidence"]:.1f}%</span>',
                    f'<span style="font-size:0.75rem;color:rgba(255,255,255,0.4);">{mod_s}</span>',
                    f'<span style="font-size:0.75rem;color:rgba(255,255,255,0.4);">{ts}</span>',
                ]):
                    with rc:
                        st.markdown(f'<div style="padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.05);">{html}</div>', unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 3: Por especie ─────────────────────────────────────────────────
    with tab_species:
        st.markdown("<br>", unsafe_allow_html=True)

        if not all_preds:
            st.markdown('<div class="alert-info">Aún no hay clasificaciones registradas.</div>', unsafe_allow_html=True)
        else:
            species_counts = Counter(p["species"] for p in all_preds)
            max_sp         = max(species_counts.values())

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title" style="margin-bottom:1.4rem;">🦅 Frecuencia por especie</div>', unsafe_allow_html=True)

            for sp, cnt in species_counts.most_common():
                common   = COMMON_NAMES.get(sp, "")
                pct_bar  = cnt / max_sp * 100
                pct_tot  = cnt / total_preds * 100
                sp_preds = [p for p in all_preds if p["species"] == sp]
                avg_c    = round(sum(p["confidence"] for p in sp_preds) / len(sp_preds), 1)

                st.markdown(f"""
                <div style="margin-bottom:14px; padding:1rem; background:rgba(255,255,255,0.03);
                            border-radius:10px; border:1px solid rgba(255,255,255,0.07);">
                    <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;flex-wrap:wrap;gap:8px;">
                        <div>
                            <span style="font-size:0.87rem;font-style:italic;color:#93c5fd;">{sp}</span>
                            <span style="font-size:0.75rem;color:rgba(255,255,255,0.38);margin-left:8px;">{common}</span>
                        </div>
                        <div style="display:flex;gap:16px;flex-wrap:wrap;">
                            <span style="font-size:0.8rem;color:rgba(255,255,255,0.5);">{cnt} veces · {pct_tot:.1f}% del total</span>
                            <span style="font-size:0.8rem;color:#60a5fa;font-weight:600;">Conf. media: {avg_c}%</span>
                        </div>
                    </div>
                    <div class="prob-bar-container">
                        <div class="prob-bar-fill" style="width:{pct_bar:.1f}%;background:linear-gradient(90deg,#1d4ed8,#60a5fa);"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ─── PÁGINA: GESTIÓN DE ROLES (solo administrador)
# ═══════════════════════════════════════════════════════════════════════════════

def page_roles():
    users     = load_users()
    stats     = load_stats()
    all_preds = stats["predictions"]

    st.markdown('<div class="page-title animate-in">👥 Gestión de roles</div>', unsafe_allow_html=True)
    st.markdown('<div class="display-subtitle" style="margin-bottom:2rem;">Asigna y modifica el rol de cada usuario registrado en AvisFauna</div>', unsafe_allow_html=True)

    # ── Buscador ──────────────────────────────────────────────────────────
    search = st.text_input("🔍 Buscar por nombre o correo", placeholder="Escribe para filtrar...",
                           key="roles_search")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabla de roles ────────────────────────────────────────────────────
    # Cabecera
    st.markdown("""
    <div style="display:grid; grid-template-columns:2fr 2fr 1.2fr 1fr 1.4fr 1.2fr;
                gap:0; padding:8px 14px; margin-bottom:4px;">
        <div class="label-muted">Usuario</div>
        <div class="label-muted">Correo</div>
        <div class="label-muted">Rol actual</div>
        <div class="label-muted">Clasif.</div>
        <div class="label-muted">Último acceso</div>
        <div class="label-muted">Acción</div>
    </div>
    """, unsafe_allow_html=True)

    any_shown = False
    for email_u, data in sorted(users.items(), key=lambda x: x[1].get("created_at", "")):
        name_u     = data.get("name", "Sin nombre")
        role_u     = data.get("role", "usuario")
        last_login = data.get("last_login", "")
        last_str   = last_login[:10] if last_login else "Nunca"
        user_preds = len([p for p in all_preds if p["user_email"] == email_u])

        if search and search.lower() not in name_u.lower() and search.lower() not in email_u.lower():
            continue

        any_shown = True
        av_col    = avatar_color(role_u)
        is_admin_fixed = (email_u == ADMIN_EMAIL)

        # Fila de usuario
        col_name, col_email, col_role, col_preds, col_last, col_action = st.columns([2, 2, 1.2, 1, 1.4, 1.2])

        with col_name:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:8px; padding:10px 0;">
                <div style="width:32px;height:32px;border-radius:50%;background:{av_col};
                            display:flex;align-items:center;justify-content:center;
                            font-weight:700;font-size:0.85rem;color:#fff;flex-shrink:0;">
                    {name_u[0].upper()}
                </div>
                <div style="font-size:0.85rem;color:rgba(255,255,255,0.85);font-weight:500;
                            white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                    {name_u}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_email:
            st.markdown(f"""
            <div style="font-size:0.77rem; color:rgba(255,255,255,0.42);
                        padding:14px 0; word-break:break-all;">
                {email_u}
            </div>
            """, unsafe_allow_html=True)

        with col_role:
            st.markdown(f"""
            <div style="padding:12px 0;">
                {role_badge_html(role_u)}
            </div>
            """, unsafe_allow_html=True)

        with col_preds:
            st.markdown(f"""
            <div style="font-size:0.9rem; font-weight:700; color:#93c5fd; padding:14px 0;">
                {user_preds}
            </div>
            """, unsafe_allow_html=True)

        with col_last:
            st.markdown(f"""
            <div style="font-size:0.78rem; color:rgba(255,255,255,0.45); padding:14px 0;">
                {last_str}
            </div>
            """, unsafe_allow_html=True)

        with col_action:
            if is_admin_fixed:
                st.markdown("""
                <div style="font-size:0.72rem; color:rgba(253,186,116,0.7);
                            padding:12px 0; font-style:italic;">
                    👑 fijo
                </div>
                """, unsafe_allow_html=True)
            else:
                new_role = st.selectbox(
                    label="rol",
                    options=list(ROLES.keys()),
                    index=list(ROLES.keys()).index(role_u),
                    key=f"rp_{email_u}",
                    label_visibility="collapsed"
                )
                if st.button("Guardar", key=f"rb_{email_u}", use_container_width=True):
                    users_fresh = load_users()
                    users_fresh[email_u]["role"] = new_role
                    save_users(users_fresh)
                    st.success(f"✅ Rol de **{name_u}** → **{ROLES[new_role]['label']}**")
                    st.rerun()

        # Separador entre filas
        st.markdown("""
        <div style="height:1px; background:rgba(255,255,255,0.05); margin:0 0 2px 0;"></div>
        """, unsafe_allow_html=True)

    if not any_shown:
        st.markdown('<div class="alert-info">No se encontraron usuarios con ese filtro.</div>',
                    unsafe_allow_html=True)

    # ── Resumen de distribución de roles ──────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="margin-bottom:1.2rem;">📊 Distribución de roles</div>',
                unsafe_allow_html=True)

    role_counts = Counter(u.get("role", "usuario") for u in users.values())
    total_u     = len(users)
    role_order  = ["administrador", "experto", "usuario"]
    col_cards   = st.columns(3)

    for col, r in zip(col_cards, role_order):
        cnt    = role_counts.get(r, 0)
        pct    = cnt / total_u * 100 if total_u else 0
        ri     = ROLES[r]
        colors = {"administrador": "#fca5a5", "experto": "#86efac", "usuario": "#93c5fd"}
        bar_c  = colors[r]
        with col:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.09);
                        border-radius:12px; padding:1rem; text-align:center;">
                <div style="font-size:1.6rem; margin-bottom:6px;">{ri['icon']}</div>
                <div style="font-size:1.8rem; font-weight:700; color:#fff; line-height:1;">{cnt}</div>
                <div style="font-size:0.72rem; color:rgba(255,255,255,0.42);
                            text-transform:uppercase; letter-spacing:0.07em; margin:4px 0 10px;">
                    {ri['label']}{'es' if r == 'administrador' else 's'}
                </div>
                <div style="background:rgba(255,255,255,0.07); border-radius:3px; height:5px;">
                    <div style="width:{pct:.1f}%; height:5px; border-radius:3px;
                                background:{bar_c}; opacity:0.7;"></div>
                </div>
                <div style="font-size:0.72rem; color:rgba(255,255,255,0.3); margin-top:5px;">
                    {pct:.0f}% del total
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ─── APP PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def main_app():
    ensure_admin_role()
    models_dict = load_models()

    if models_dict is None:
        st.error("⚠️ No se pudieron cargar los modelos. Verifica la configuración de Google Drive.")
        return

    model_choice = render_sidebar(models_dict)
    current_page = st.session_state.get("current_page", "clasificador")
    role         = st.session_state.get("user_role", "usuario")

    if current_page == "clasificador":
        page_clasificador(models_dict, model_choice)

    elif current_page == "estadisticas":
        if can(role, "ver_estadisticas"):
            page_estadisticas()
        else:
            st.error("⛔ No tienes permiso para ver esta sección.")

    elif current_page == "dashboard":
        if can(role, "dashboard_admin"):
            page_dashboard_admin()
        else:
            st.error("⛔ Acceso restringido a administradores.")

    elif current_page == "roles":
        if can(role, "gestionar_usuarios"):
            page_roles()
        else:
            st.error("⛔ Acceso restringido a administradores.")


# ═══════════════════════════════════════════════════════════════════════════════
# ─── PERSISTENCIA DE SESIÓN (archivo local)
# ═══════════════════════════════════════════════════════════════════════════════

SESSION_FILE    = ".session_store.json"
INACTIVITY_MINS = 15   # minutos sin actividad → cierre automático

def save_session(email, name, role):
    """Guarda la sesión en disco para sobrevivir recargas."""
    now = datetime.now().isoformat()
    data = {
        "email":       email,
        "name":        name,
        "role":        role,
        "last_active": now,
        "login_time":  now
    }
    with open(SESSION_FILE, "w") as f:
        json.dump(data, f)

def load_session():
    """Carga sesión del disco. Retorna None si no existe o expiró."""
    if not os.path.exists(SESSION_FILE):
        return None
    try:
        with open(SESSION_FILE, "r") as f:
            data = json.load(f)
        last = datetime.fromisoformat(data["last_active"])
        if datetime.now() - last > timedelta(minutes=INACTIVITY_MINS):
            clear_session()
            return None
        # Garantizar campo login_time por compatibilidad con sesiones antiguas
        if "login_time" not in data:
            data["login_time"] = data["last_active"]
        return data
    except Exception:
        return None

def update_session_activity():
    """Actualiza el timestamp de última actividad."""
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, "r") as f:
                data = json.load(f)
            data["last_active"] = datetime.now().isoformat()
            with open(SESSION_FILE, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

def clear_session():
    """Elimina el archivo de sesión."""
    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)

def seconds_until_timeout():
    """Devuelve los segundos que quedan antes del cierre por inactividad."""
    if not os.path.exists(SESSION_FILE):
        return 0
    try:
        with open(SESSION_FILE, "r") as f:
            data = json.load(f)
        last    = datetime.fromisoformat(data["last_active"])
        elapsed = (datetime.now() - last).total_seconds()
        return max(0, INACTIVITY_MINS * 60 - elapsed)
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
# ─── PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Restaurar sesión desde disco si session_state está vacío ────────
    if "authenticated" not in st.session_state:
        saved = load_session()
        if saved:
            st.session_state.authenticated = True
            st.session_state.user_email    = saved["email"]
            st.session_state.user_name     = saved["name"]
            st.session_state.user_role     = saved["role"]
            st.session_state.login_time    = saved.get("login_time", datetime.now().isoformat())
            st.session_state.current_page  = st.session_state.get("current_page", "clasificador")
        else:
            st.session_state.authenticated = False

    # ── 2. Verificar inactividad si hay sesión activa ──────────────────────
    if st.session_state.get("authenticated"):
        secs_left = seconds_until_timeout()
        if secs_left <= 0:
            # Tiempo agotado → cerrar sesión
            clear_session()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.authenticated   = False
            st.session_state.session_expired = True
            st.rerun()
        else:
            # Registrar actividad en cada interacción
            update_session_activity()

            # ── Banner de cuenta regresiva si quedan ≤ 3 minutos ──────────
            if secs_left <= 180:
                mins = int(secs_left // 60)
                secs = int(secs_left % 60)
                st.markdown(f"""
                <div style="position:fixed; bottom:1.2rem; right:1.4rem; z-index:9999;
                            background:rgba(180,100,0,0.92); border:1px solid rgba(253,186,116,0.5);
                            border-radius:12px; padding:10px 18px; display:flex; align-items:center;
                            gap:10px; backdrop-filter:blur(8px); box-shadow:0 4px 20px rgba(0,0,0,0.4);">
                    <span style="font-size:1.1rem;">⏱️</span>
                    <div>
                        <div style="font-size:0.78rem; color:rgba(255,255,255,0.65); line-height:1;">Sesión expira en</div>
                        <div style="font-size:1rem; font-weight:700; color:#fff; line-height:1.3;">
                            {mins}:{secs:02d}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── 3. Mostrar aviso de sesión expirada ───────────────────────────────
    if st.session_state.get("session_expired"):
        st.warning("⏱ Tu sesión expiró por inactividad. Por favor inicia sesión de nuevo.")
        del st.session_state["session_expired"]

    # ── 4. Routing principal ───────────────────────────────────────────────
    if not st.session_state.get("authenticated"):
        auth_screen()
    else:
        main_app()

if __name__ == "__main__":
    main()