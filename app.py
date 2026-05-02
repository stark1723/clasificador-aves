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

# ─── CONFIGURACIÓN DE PÁGINA ────────────────────────────────────────────────
st.set_page_config(
    page_title="AvisFauna — Identificador de Aves",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── ESTILOS GLOBALES ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Fondo general ── */
.stApp {
    background: linear-gradient(135deg, #0a1628 0%, #0f2240 50%, #0d1f35 100%);
    min-height: 100vh;
}

/* ── Ocultar elementos Streamlit por defecto ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* ── Tarjetas ── */
.card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 16px;
    padding: 2rem;
    backdrop-filter: blur(12px);
}
.card-elevated {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 20px;
    padding: 2.5rem;
    backdrop-filter: blur(16px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* ── Tipografía ── */
.display-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.2;
    letter-spacing: -0.02em;
}
.display-subtitle {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.55);
    font-weight: 300;
    letter-spacing: 0.01em;
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-weight: 600;
    color: #ffffff;
}
.label-muted {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.45);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}

/* ── Badges de especie ── */
.species-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1a4a8a, #2563c0);
    color: #93c5fd;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid rgba(147,197,253,0.25);
    margin: 3px 3px 3px 0;
    letter-spacing: 0.02em;
}

/* ── Resultado principal ── */
.result-species {
    font-family: 'Playfair Display', serif;
    font-size: 1.7rem;
    font-weight: 600;
    color: #60a5fa;
    font-style: italic;
    margin: 0.5rem 0;
}
.result-confidence {
    font-size: 2.5rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1;
}
.result-label {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.45);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Barra de probabilidad personalizada ── */
.prob-bar-container {
    background: rgba(255,255,255,0.06);
    border-radius: 4px;
    height: 6px;
    width: 100%;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.8s ease;
}

/* ── Auth card ── */
.auth-container {
    max-width: 480px;
    margin: 2rem auto;
}
.auth-logo {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #ffffff;
    text-align: center;
    margin-bottom: 0.3rem;
}
.auth-tagline {
    text-align: center;
    color: rgba(255,255,255,0.45);
    font-size: 0.9rem;
    margin-bottom: 2rem;
}
.auth-divider {
    height: 1px;
    background: rgba(255,255,255,0.1);
    margin: 1.5rem 0;
}

/* ── Inputs Streamlit → estilo oscuro ── */
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
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1.5rem !important;
    transition: all 0.2s !important;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1e40af, #1d4ed8) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(37,99,235,0.35) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(6, 18, 40, 0.92) !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
    color: rgba(255,255,255,0.75) !important;
}

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

/* ── Mensajes de estado ── */
.stSuccess, .stError, .stWarning, .stInfo {
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #2563eb, #60a5fa) !important;
    border-radius: 4px !important;
}

/* ── Animación de entrada ── */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.animate-in {
    animation: fadeSlideIn 0.5s ease forwards;
}

/* ── Icono de ave decorativo ── */
.bird-icon {
    font-size: 3rem;
    display: block;
    text-align: center;
    margin-bottom: 0.5rem;
    filter: drop-shadow(0 0 20px rgba(96,165,250,0.4));
}

/* ── Chip de modelo ── */
.model-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(96,165,250,0.12);
    border: 1px solid rgba(96,165,250,0.25);
    color: #93c5fd;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 4px 10px;
    border-radius: 6px;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    color: rgba(255,255,255,0.55) !important;
    font-family: 'DM Sans', sans-serif !important;
    border-radius: 8px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(37,99,235,0.35) !important;
    color: #ffffff !important;
}

/* ── Selectbox ── */
.stSelectbox > div {
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)


# ─── BASE DE DATOS DE USUARIOS (JSON local) ──────────────────────────────────

USERS_FILE = "users_db.json"
RESET_TOKENS_FILE = "reset_tokens.json"

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

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def validate_email(email):
    return re.match(r'^[\w\.-]+@[\w\.-]+\.\w{2,}$', email)

def generate_token(length=32):
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


# ─── CONFIGURACIÓN DE MODELOS ────────────────────────────────────────────────

GDRIVE_MODEL_1_ID = '1FcU2jJEYqU0c971n9WIytWfFaTD018z-'
GDRIVE_MODEL_2_ID = '1ZvA60kdraCRDw8wGx83FqfUkwWOeykPC'
MODEL_1_PATH = 'inception_finetuned.h5'
MODEL_2_PATH  = 'VGG16_finetuned.h5'
IMG_SIZE = (224, 224)

CLASSES = [
    'Anhinga anhinga',
    'Butorides striata',
    'Chamaepetes goudotii',
    'Colinus cristatus',
    'Nycticorax nycticorax',
    'Ortalis guttata',
    'Penelope montagnii',
    'Podilymbus podiceps',
    'Tachybaptus dominicus',
    'Tigrisoma lineatum'
]

COMMON_NAMES = {
    'Anhinga anhinga':      'Pato aguja',
    'Butorides striata':    'Garza verdosa',
    'Chamaepetes goudotii': 'Pava caucana',
    'Colinus cristatus':    'Codorniz crestada',
    'Nycticorax nycticorax':'Garza nocturna',
    'Ortalis guttata':      'Guacharaca moteada',
    'Penelope montagnii':   'Pava andina',
    'Podilymbus podiceps':  'Zambullidor piquipinto',
    'Tachybaptus dominicus':'Zambullidor menor',
    'Tigrisoma lineatum':   'Garza tigre rayada'
}


# ─── DESCARGA Y CARGA DE MODELOS ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def download_model(file_id, output_path):
    if not os.path.exists(output_path):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_path, quiet=False)
    return os.path.exists(output_path)

@st.cache_resource(show_spinner=False)
def load_models():
    try:
        ok1 = download_model(GDRIVE_MODEL_1_ID, MODEL_1_PATH)
        ok2 = download_model(GDRIVE_MODEL_2_ID, MODEL_2_PATH)
        if not ok1 or not ok2:
            return None
        m1 = tf.keras.models.load_model(MODEL_1_PATH, compile=False)
        m2 = tf.keras.models.load_model(MODEL_2_PATH, compile=False)
        m1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        m2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return {"Inception V3": m1, "VGG16": m2}
    except Exception as e:
        st.error(f"Error al cargar los modelos: {e}")
        return None


# ─── PREDICCIÓN ──────────────────────────────────────────────────────────────

def predict(image, model):
    img = image.convert('RGB').resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds = model.predict(arr)
    idx = int(np.argmax(preds, axis=1)[0])
    return CLASSES[idx], float(preds[0][idx]) * 100, preds[0]


# ─── PANTALLA DE AUTENTICACIÓN ───────────────────────────────────────────────

def auth_screen():
    st.markdown('<div class="auth-container animate-in">', unsafe_allow_html=True)

    # Logo
    st.markdown("""
    <div style="text-align:center; margin-bottom: 2rem; padding-top: 1rem;">
        <span class="bird-icon">🦅</span>
        <div class="auth-logo">AvisFauna</div>
        <div class="auth-tagline">Identificación inteligente de aves colombianas</div>
    </div>
    """, unsafe_allow_html=True)

    tab_login, tab_register, tab_reset = st.tabs(["Iniciar sesión", "Registrarse", "Olvidé mi contraseña"])

    # ── Login ──────────────────────────────────────────────────────────────
    with tab_login:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.form("form_login"):
            email = st.text_input("Correo electrónico", placeholder="usuario@correo.com")
            password = st.text_input("Contraseña", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Entrar →", use_container_width=True)

        if submitted:
            if not email or not password:
                st.error("Por favor completa todos los campos.")
            else:
                users = load_users()
                h = hash_password(password)
                if email in users and users[email]["password"] == h:
                    st.session_state.authenticated = True
                    st.session_state.user_email = email
                    st.session_state.user_name  = users[email].get("name", email.split("@")[0])
                    st.success(f"¡Bienvenido/a, {st.session_state.user_name}!")
                    st.rerun()
                else:
                    st.error("Correo o contraseña incorrectos.")

    # ── Registro ───────────────────────────────────────────────────────────
    with tab_register:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.form("form_register"):
            name     = st.text_input("Nombre completo", placeholder="Ana García")
            reg_email = st.text_input("Correo electrónico", placeholder="usuario@correo.com")
            reg_pass  = st.text_input("Contraseña", type="password", placeholder="Mínimo 8 caracteres")
            reg_pass2 = st.text_input("Confirmar contraseña", type="password", placeholder="Repite tu contraseña")
            submitted_r = st.form_submit_button("Crear cuenta →", use_container_width=True)

        if submitted_r:
            users = load_users()
            ok = True
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
                users[reg_email] = {
                    "name": name,
                    "password": hash_password(reg_pass),
                    "created_at": datetime.now().isoformat()
                }
                save_users(users)
                st.success("✅ Cuenta creada. ¡Ya puedes iniciar sesión!")

    # ── Recuperar contraseña ───────────────────────────────────────────────
    with tab_reset:
        st.markdown("<br>", unsafe_allow_html=True)

        # Paso 1: solicitar token
        if "reset_step" not in st.session_state:
            st.session_state.reset_step = 1

        if st.session_state.reset_step == 1:
            st.markdown('<p class="display-subtitle" style="text-align:center; margin-bottom:1rem;">Ingresa tu correo y te daremos un código de recuperación.</p>', unsafe_allow_html=True)
            with st.form("form_reset_req"):
                reset_email = st.text_input("Correo electrónico", placeholder="usuario@correo.com")
                sub_req = st.form_submit_button("Enviar código →", use_container_width=True)

            if sub_req:
                users = load_users()
                if not reset_email:
                    st.error("Ingresa tu correo.")
                elif not validate_email(reset_email):
                    st.error("Formato de correo inválido.")
                elif reset_email not in users:
                    st.error("No encontramos una cuenta con ese correo.")
                else:
                    token = generate_token(8).upper()
                    tokens = load_reset_tokens()
                    tokens[reset_email] = {
                        "token": token,
                        "expires": (datetime.now() + timedelta(minutes=15)).isoformat()
                    }
                    save_reset_tokens(tokens)
                    st.session_state.reset_email_target = reset_email
                    st.session_state.reset_step = 2
                    # En producción enviarías el token por email.
                    # Aquí lo mostramos directamente (demo):
                    st.info(f"🔑 Tu código de recuperación (demo): **{token}**\n\nVálido por 15 minutos.")
                    st.rerun()

        elif st.session_state.reset_step == 2:
            target_email = st.session_state.get("reset_email_target", "")
            st.markdown(f'<p style="color:rgba(255,255,255,0.55); text-align:center; font-size:0.9rem;">Código enviado a: <strong style="color:#93c5fd;">{target_email}</strong></p>', unsafe_allow_html=True)
            with st.form("form_reset_verify"):
                input_token  = st.text_input("Código de verificación", placeholder="XXXXXXXX")
                new_pass     = st.text_input("Nueva contraseña", type="password", placeholder="Mínimo 8 caracteres")
                new_pass2    = st.text_input("Confirmar contraseña", type="password", placeholder="Repite tu nueva contraseña")
                sub_verify   = st.form_submit_button("Restablecer contraseña →", use_container_width=True)

            col_b, _ = st.columns([1, 3])
            with col_b:
                if st.button("← Volver", use_container_width=True):
                    st.session_state.reset_step = 1
                    st.rerun()

            if sub_verify:
                tokens = load_reset_tokens()
                ok = True
                if not input_token or not new_pass or not new_pass2:
                    st.error("Completa todos los campos."); ok = False
                elif target_email not in tokens:
                    st.error("No hay un código activo para este correo."); ok = False
                else:
                    t_data = tokens[target_email]
                    expired = datetime.now() > datetime.fromisoformat(t_data["expires"])
                    if expired:
                        st.error("El código ha expirado. Solicita uno nuevo."); ok = False
                    elif input_token.upper() != t_data["token"]:
                        st.error("Código incorrecto."); ok = False

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
                        st.success("✅ Contraseña restablecida. ¡Ya puedes iniciar sesión!")
                        st.session_state.reset_step = 1
                        del st.session_state["reset_email_target"]
                        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ─── SIDEBAR (app principal) ──────────────────────────────────────────────────

def render_sidebar(models_dict):
    with st.sidebar:
        st.markdown(f"""
        <div style="padding: 1rem 0 1.5rem 0;">
            <div style="font-family:'Playfair Display',serif; font-size:1.4rem; font-weight:700; color:#fff;">
                🦅 AvisFauna
            </div>
            <div style="font-size:0.78rem; color:rgba(255,255,255,0.4); margin-top:4px;">
                Identificación inteligente
            </div>
        </div>
        <div style="height:1px; background:rgba(255,255,255,0.08); margin-bottom:1.5rem;"></div>
        <div style="font-size:0.78rem; color:rgba(255,255,255,0.45); text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.6rem;">
            Sesión activa
        </div>
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:1.5rem;">
            <div style="width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,#1d4ed8,#60a5fa);
                        display:flex;align-items:center;justify-content:center;font-weight:600;font-size:0.9rem;color:#fff;">
                {st.session_state.user_name[0].upper()}
            </div>
            <div>
                <div style="color:#fff; font-size:0.9rem; font-weight:500;">{st.session_state.user_name}</div>
                <div style="color:rgba(255,255,255,0.4); font-size:0.75rem;">{st.session_state.user_email}</div>
            </div>
        </div>
        <div style="height:1px; background:rgba(255,255,255,0.08); margin-bottom:1.5rem;"></div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="label-muted" style="margin-bottom:0.5rem;">Modelo de clasificación</div>', unsafe_allow_html=True)
        model_choice = st.selectbox(
            label="Modelo",
            options=list(models_dict.keys()),
            label_visibility="collapsed"
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="height:1px; background:rgba(255,255,255,0.08); margin-bottom:1.5rem;"></div>
        <div class="label-muted" style="margin-bottom:0.8rem;">Especies detectables</div>
        """, unsafe_allow_html=True)

        for sci in CLASSES:
            common = COMMON_NAMES.get(sci, "")
            st.markdown(f"""
            <div style="margin-bottom:8px;">
                <div style="font-size:0.82rem; color:rgba(255,255,255,0.7); font-style:italic;">{sci}</div>
                <div style="font-size:0.74rem; color:rgba(255,255,255,0.38);">{common}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="height:1px; background:rgba(255,255,255,0.08); margin: 1.5rem 0;"></div>
        """, unsafe_allow_html=True)

        if st.button("Cerrar sesión", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    return model_choice


# ─── PANTALLA PRINCIPAL DE LA APP ────────────────────────────────────────────

def main_app():
    models_dict = None
    with st.spinner("Cargando modelos de IA..."):
        models_dict = load_models()

    if models_dict is None:
        st.error("⚠️ No se pudieron cargar los modelos. Verifica la configuración de Google Drive.")
        return

    model_choice = render_sidebar(models_dict)

    # ── Encabezado ──────────────────────────────────────────────────────────
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
            <div class="model-chip">
                <span>🧠</span> {model_choice}
            </div>
            <div style="font-size:0.8rem; color:rgba(255,255,255,0.35); margin-top:8px;">
                {len(CLASSES)} especies · Colombia
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Zona principal ──────────────────────────────────────────────────────
    col_upload, col_result = st.columns([1, 1], gap="large")

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
            st.image(image, caption="Imagen cargada", use_column_width=True,
                     output_format="PNG")
            st.markdown(f"""
            <div style="display:flex; gap:10px; margin-top:0.8rem; flex-wrap:wrap;">
                <span style="font-size:0.78rem; color:rgba(255,255,255,0.4);">
                    {image.width} × {image.height} px
                </span>
                <span style="font-size:0.78rem; color:rgba(255,255,255,0.4);">·</span>
                <span style="font-size:0.78rem; color:rgba(255,255,255,0.4);">
                    {uploaded_file.name}
                </span>
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

    with col_result:
        st.markdown('<div class="card animate-in" style="min-height:320px;">', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="margin-bottom:1.2rem;">🔬 Resultado</div>', unsafe_allow_html=True)

        if uploaded_file:
            if st.button("✦ Identificar especie", use_container_width=True):
                with st.spinner("Analizando imagen..."):
                    try:
                        model = models_dict[model_choice]
                        species, confidence, all_preds = predict(image, model)
                        common = COMMON_NAMES.get(species, "—")

                        st.session_state.last_result = {
                            "species": species,
                            "common": common,
                            "confidence": confidence,
                            "all_preds": all_preds.tolist()
                        }
                    except Exception as e:
                        st.error(f"Error durante la predicción: {e}")

        if "last_result" in st.session_state and uploaded_file:
            r = st.session_state.last_result

            st.markdown(f"""
            <div style="margin-bottom:1.5rem;">
                <div class="result-label">Especie identificada</div>
                <div class="result-species">{r['species']}</div>
                <div style="font-size:0.9rem; color:rgba(255,255,255,0.5); margin-top:2px;">
                    {r['common']}
                </div>
            </div>
            <div style="display:flex; align-items:baseline; gap:8px; margin-bottom:1.5rem;">
                <div>
                    <div class="result-label">Confianza</div>
                    <div class="result-confidence">{r['confidence']:.1f}%</div>
                </div>
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

        r = st.session_state.last_result
        sorted_pairs = sorted(
            zip(CLASSES, r["all_preds"]),
            key=lambda x: x[1],
            reverse=True
        )

        cols = st.columns(2)
        for i, (cls, prob) in enumerate(sorted_pairs):
            pct = prob * 100
            common = COMMON_NAMES.get(cls, "")
            is_top = (cls == r["species"])
            color = "#60a5fa" if is_top else "#475569"

            with cols[i % 2]:
                st.markdown(f"""
                <div style="margin-bottom:14px;">
                    <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:5px;">
                        <div>
                            <span style="font-size:0.82rem; font-style:italic; color:{'#60a5fa' if is_top else 'rgba(255,255,255,0.65)'};">
                                {cls}
                            </span>
                            <span style="font-size:0.7rem; color:rgba(255,255,255,0.35); margin-left:6px;">
                                {common}
                            </span>
                        </div>
                        <span style="font-size:0.82rem; font-weight:600; color:{'#93c5fd' if is_top else 'rgba(255,255,255,0.5)'};">
                            {pct:.1f}%
                        </span>
                    </div>
                    <div class="prob-bar-container">
                        <div class="prob-bar-fill" style="width:{pct:.1f}%; background:{color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ─── PUNTO DE ENTRADA ─────────────────────────────────────────────────────────

def main():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        auth_screen()
    else:
        main_app()

if __name__ == "__main__":
    main()
