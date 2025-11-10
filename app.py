import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess

# ===========================================================
# CONFIGURACIÓN GENERAL
# ===========================================================
st.set_page_config(
    page_title="Detección de Aves 🦜",
    page_icon="🦜",
    layout="wide",
)

# ===========================================================
# ESTILO PERSONALIZADO
# ===========================================================
st.markdown(
    """
<style>
/* Fondo principal degradado verde a azul */
.stApp {
    background: linear-gradient(180deg, #80ba26 0%, #00abc8 100%);
}

/* Rectángulo superior blanco */
.top-bar {
    background-color: white;
    border-radius: 20px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* Títulos dentro del rectángulo */
.top-title {
    font-size: 1.3rem;
    font-weight: 800;
    color: #1A1A1A;
    display: flex;
    align-items: center;
}

/* Emoji al lado del título */
.top-title span {
    margin-right: 10px;
}

/* Contenedor principal (tarjetas) */
.block-container {
    background-color: rgba(255, 255, 255, 0.1);
    padding: 1.5rem 1.5rem 3rem 1.5rem;
}

/* Tarjetas */
.card {
    background-color: #FFFFFF;
    border-radius: 18px;
    padding: 1.5rem;
    box-shadow: 0 6px 14px rgba(0,0,0,0.1);
}

/* Botones */
.stButton>button {
    background-color: #00abc8;
    color: white;
    font-weight: 700;
    border-radius: 999px;
    padding: 0.6rem 1.4rem;
    border: none;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    background-color: #80ba26;
    color: white;
}

/* Selectbox y FileUploader */
.stSelectbox > div > div,
.stFileUploader > div {
    border-radius: 12px;
}

/* Alertas */
.stAlert {
    border-radius: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ===========================================================
# CONFIGURACIÓN DE MODELOS
# ===========================================================
MODEL_CONFIG = {
    "EfficientNet B0": {
        "path": "modelos/efficenet.keras",
        "input_size": (224, 224),
        "preprocess": eff_preprocess,
    },
    "VGG16": {
        "path": "modelos/vgg16.keras",
        "input_size": (224, 224),
        "preprocess": vgg_preprocess,
    },
}

CLASS_NAMES_PATH = "class_names.txt"


@st.cache_resource(show_spinner="Cargando modelo seleccionado…")
def cargar_modelo(nombre_modelo: str):
    """Carga modelo .keras con safe_mode desactivado"""
    config = MODEL_CONFIG.get(nombre_modelo)
    if config is None:
        raise ValueError(f"No existe configuración para el modelo: {nombre_modelo}")

    path = config["path"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el modelo en: {path}")

    model = tf.keras.models.load_model(path, safe_mode=False, compile=False)
    return model


@st.cache_data
def cargar_clases(path: str):
    """Lee class_names.txt (una clase por línea)"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de clases en: {path}")
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines


# ===========================================================
# CABECERA
# ===========================================================
st.markdown("<h1 style='text-align:center; color:white;'>🦜 Detección de Aves</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:white; font-size:1.1rem;'>"
    "Proyecto con modelos de Deep Learning para clasificación de aves."
    "</p>",
    unsafe_allow_html=True,
)
st.write("")

# ===========================================================
# LAYOUT PRINCIPAL
# ===========================================================
col_left, col_right = st.columns([1, 1])

# ------------------------ COLUMNA IZQUIERDA -----------------
with col_left:
    st.markdown('<div class="top-bar"><div class="top-title"><span>⚙️</span>Configuración del modelo</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    modelo_seleccionado = st.selectbox(
        "Selecciona el modelo:",
        list(MODEL_CONFIG.keys()),
        index=0,
    )

    # Mensaje de modelo cargado exitosamente
    try:
        modelo = cargar_modelo(modelo_seleccionado)
        st.success(f"✅ Modelo **{modelo_seleccionado}** cargado exitosamente.")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        modelo = None

    st.markdown("---")

    st.subheader("📤 Cargar imagen")
    archivo_imagen = st.file_uploader(
        "Sube una imagen de un ave (JPG o PNG):",
        type=["jpg", "jpeg", "png"],
    )

    imagen = None
    if archivo_imagen is not None:
        imagen = Image.open(archivo_imagen).convert("RGB")
        st.image(imagen, caption="Imagen cargada correctamente ✅", use_column_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------ COLUMNA DERECHA -------------------
with col_right:
    st.markdown('<div class="top-bar"><div class="top-title"><span>📊</span>Resultados de la predicción</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    pred_df = None

    try:
        class_names = cargar_clases(CLASS_NAMES_PATH)
    except Exception as e:
        st.error(f"Error al cargar las clases: {e}")
        class_names = None

    if st.button("🔍 Clasificar ave", use_container_width=True):
        if modelo is None:
            st.warning("Primero selecciona y carga un modelo válido.")
        elif imagen is None:
            st.warning("Sube una imagen para analizar.")
        elif class_names is None:
            st.warning("No se pudieron cargar las clases.")
        else:
            try:
                with st.spinner(f"Realizando predicción con {modelo_seleccionado}…"):
                    config = MODEL_CONFIG[modelo_seleccionado]
                    preprocess_fun = config.get("preprocess", lambda x: x / 255.0)
                    input_size = config["input_size"]

                    img_resized = imagen.resize(input_size)
                    img_array = np.array(img_resized, dtype=np.float32)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_fun(img_array.copy())

                    preds = modelo.predict(img_array)[0]
                    n = min(len(preds), len(class_names))
                    preds, clases = preds[:n], class_names[:n]

                    pred_df = pd.DataFrame({"Especie": clases, "Probabilidad": preds}).sort_values("Probabilidad", ascending=False)
                    top_row = pred_df.iloc[0]

                    st.success(f"**Ave predicha:** {top_row['Especie']} con probabilidad {top_row['Probabilidad']*100:.2f}%")
                    st.write("### 🔢 Probabilidades por especie")
                    st.dataframe(pred_df.style.format({"Probabilidad": "{:.4f}"}), use_container_width=True)

                    st.write("### 📈 Top 5 clases (gráfico)")
                    top5 = pred_df.head(5).set_index("Especie")
                    st.bar_chart(top5)
            except Exception as e:
                st.error(f"Error al realizar la predicción: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
