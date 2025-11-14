import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import gdown

# --- CONFIGURACIÓN DE GOOGLE DRIVE ---

# IDs de los archivos en Google Drive
GDRIVE_MODEL_1_ID = '1FcU2jJEYqU0c971n9WIytWfFaTD018z-'  # inception_finetuned.h5
GDRIVE_MODEL_2_ID = '1ZvA60kdraCRDw8wGx83FqfUkwWOeykPC'  # VGG16_finetuned.h5

# Rutas locales donde se guardarán los modelos descargados
MODEL_1_PATH = 'inception_finetuned.h5'
MODEL_2_PATH = 'VGG16_finetuned.h5'

# Lista de clases para cada modelo (orden correcto según pruebas)
CLASSES_MODEL_1 = [
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

CLASSES_MODEL_2 = [
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

# Dimensiones de entrada que esperan tus modelos (ej. 224x224)
IMG_SIZE = (224, 224) 

# --- FUNCIÓN PARA DESCARGAR MODELOS DESDE GOOGLE DRIVE ---

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

# --- FUNCIÓN DE CARGA DE MODELOS ---

@st.cache_resource
def load_models():
    """Descarga y carga ambos modelos pre-entrenados."""
    try:
        # Descargar modelos si no existen
        if not download_model_from_drive(GDRIVE_MODEL_1_ID, MODEL_1_PATH):
            return None
        if not download_model_from_drive(GDRIVE_MODEL_2_ID, MODEL_2_PATH):
            return None
        
        # Cargar modelos con compile=False para evitar problemas de compatibilidad
        model1 = tf.keras.models.load_model(MODEL_1_PATH, compile=False)
        model2 = tf.keras.models.load_model(MODEL_2_PATH, compile=False)
        
        # Recompilar los modelos manualmente
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

# --- FUNCIÓN DE PREPROCESAMIENTO Y PREDICCIÓN ---

def preprocess_and_predict(image, model_key, models_dict):
    """Prepara la imagen y realiza la predicción."""
    
    # 1. Obtener el modelo y las clases
    model = models_dict[model_key]
    if "Modelo 1" in model_key:
        classes = CLASSES_MODEL_1
    else:
        classes = CLASSES_MODEL_2

    # 2. Preprocesamiento (Asegúrate de que esto coincide con el entrenamiento)
    img = image.convert('RGB')  # Asegurar que la imagen esté en RGB
    img = img.resize(IMG_SIZE)  # Redimensionar
    img_array = np.array(img) / 255.0  # Normalizar (si tu modelo lo requiere)
    img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión de batch

    # 3. Predicción
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = classes[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100

    return predicted_class, confidence, predictions[0]

# --- INTERFAZ DE STREAMLIT ---

def main():
    st.set_page_config(page_title="Clasificador de Aves", page_icon="🦅", layout="wide")
    
    st.title("🦅 Clasificador de Aves")
    st.markdown("Sube una imagen de un ave y selecciona el modelo para predecir su especie.")
    
    st.sidebar.header("Opciones")
    
    models_dict = load_models()
    if models_dict is None:
        st.warning("⚠️ No se pudieron cargar los modelos. Por favor, verifica la configuración.")
        return  # Salir si la carga de modelos falló

    # Selector de modelo en la barra lateral
    model_choice = st.sidebar.selectbox(
        "**Selecciona el Modelo a Utilizar:**",
        list(models_dict.keys())
    )

    # Información adicional en la barra lateral
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Acerca de los modelos")
    if "Modelo 1" in model_choice:
        st.sidebar.write(f"**Clases:** {', '.join(CLASSES_MODEL_1)}")
    else:
        st.sidebar.write(f"**Clases:** {', '.join(CLASSES_MODEL_2)}")

    st.subheader("1. Sube tu Imagen")
    uploaded_file = st.file_uploader("Elige un archivo JPG/PNG...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Mostrar la imagen subida
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption='Imagen de Ave Subida', use_column_width=True)
        
        with col2:
            st.subheader(f"2. Predicción usando: **{model_choice}**")
            
            # Botón de predicción
            if st.button('🔍 Obtener Predicción', type="primary"):
                with st.spinner('Realizando la predicción...'):
                    try:
                        predicted_class, confidence, all_predictions = preprocess_and_predict(image, model_choice, models_dict)
                        
                        st.success(f"✅ Predicción completada")
                        st.markdown(f"### **Especie Predicha:** {predicted_class}")
                        st.markdown(f"### **Confianza:** {confidence:.2f}%")
                        
                        # Mostrar barra de progreso
                        st.progress(confidence / 100)
                        
                        # Mostrar todas las probabilidades
                        st.markdown("---")
                        st.markdown("**Probabilidades por clase:**")
                        classes = CLASSES_MODEL_1 if "Modelo 1" in model_choice else CLASSES_MODEL_2
                        for cls, prob in zip(classes, all_predictions):
                            st.write(f"- {cls}: {prob * 100:.2f}%")

                    except Exception as e:
                        st.error(f"Ocurrió un error durante la predicción: {e}")

if __name__ == "__main__":
    main()