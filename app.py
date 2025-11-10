import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# ===========================================================
# CONFIGURACIÓN GENERAL DE LA PÁGINA
# ===========================================================
st.set_page_config(
    page_title="Detección de Aves 🦜",
    page_icon="🦜",
    layout="wide",
)

# ===========================================================
# ESTILO PERSONALIZADO (franja vino tinto arriba + amarillo)
# ===========================================================
st.markdown(
    """
<style>
/* Fondo general con franja vino tinto arriba y resto amarillo */
.stApp {
    background: linear-gradient(
        180deg,
        #6D090D 0%,
        #6D090D 30%,
        #FCDD09 30%,
        #FFF9C4 100%
    );
}

/* Contenedor principal */
.block-container {
    background-color: rgba(0,0,0,0.00);
    padding: 1.5rem 1.5rem 3rem 1.5rem;
}

/* Título principal */
h1 {
    color: #FFFFFF !important;
    text-align: center;
    font-weight: 800;
}

/* Subtítulos */
h2, h3, h4 {
    color: #1A1A1A !important;
}

/* Tarjetas blancas redondeadas */
.card {
    background-color: #FFFFFF;
    border-radius: 18px;
    padding: 1.5rem;
    box-shadow: 0 8px 18px rgba(0,0,0,0.12);
}

/* Botones principales */
.stButton>button {
    background-color: #6D090D;
    color: #FFFFFF;
    font-weight: 700;
    border-radius: 999px;
    padding: 0.6rem 1.4rem;
    border: none;
}

.stButton>button:hover {
    background-color: #8c1015;
    color: #FFFFFF;
}

/* Selectbox y file uploader */
.stSelectbox > div > div,
.stFileUploader > div {
    border-radius: 999px;
}

/* Mensajes de éxito */
.stAlert {
    border-radius: 16px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ===========================================================
# CONFIGURACIÓN DE MODELOS
# ===========================================================
# Ajusta las rutas a los nombres REALES de tus archivos .keras
MODEL_CONFIG = {
    "EfficientNet B0": {
        "path": "modelos/efficenet.keras",
        "input_size": (224, 224),
    },
    "VGG16": {
        "path": "modelos/vgg16.keras",
        "input_size": (224, 224),
    },
    # Si tienes otro modelo, lo agregas aquí:
    # "Otro modelo": {"path": "modelos/otro_modelo.keras", "input_size": (224, 224)},
}

CLASS_NAMES_PATH = "class_names.txt"


@st.cache_resource(show_spinner="Cargando modelo seleccionado…")
def cargar_modelo(nombre_modelo: str):
    """
    Carga el modelo .keras usando safe_mode=False (para evitar errores
    de capas que reciben múltiples tensores) y sin compilar (solo inferencia).
    """
    config = MODEL_CONFIG.get(nombre_modelo)
    if config is None:
        raise ValueError(f"No existe configuración para el modelo: {nombre_modelo}")

    model_path = config["path"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

    # Carga tolerante para modelos entrenados en versiones anteriores
    model = tf.keras.models.load_model(
        model_path,
        safe_mode=False,
        compile=False,
    )
    return model


@st.cache_data
def cargar_clases(path: str):
    """
    Lee el archivo class_names.txt (una clase por línea).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de clases en: {path}")
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines


# ===========================================================
# CABECERA
# ===========================================================
st.markdown("<h1>🦜 Detección de Aves</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#FFFFFF; font-size:1.05rem;'>"
    "Proyecto con dos modelos de Deep Learning para clasificación de aves."
    "</p>",
    unsafe_allow_html=True,
)

st.write("")  # pequeño espacio

# ===========================================================
# LAYOUT PRINCIPAL
# ===========================================================
col_left, col_right = st.columns([1.1, 1.1])

# ------------------------ COLUMNA IZQUIERDA -----------------
with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("⚙️ Configuración del modelo")

    modelo_seleccionado = st.selectbox(
        "Selecciona el modelo:",
        list(MODEL_CONFIG.keys()),
        index=0,
    )

    st.markdown("---")

    st.subheader("📤 Cargar imagen")
    archivo_imagen = st.file_uploader(
        "Sube una imagen de un ave (JPG o PNG):",
        type=["jpg", "jpeg", "png"],
    )

    imagen = None
    if archivo_imagen is not None:
        imagen = Image.open(archivo_imagen).convert("RGB")
        st.image(
            imagen,
            caption="Imagen cargada correctamente ✅",
            use_column_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------ COLUMNA DERECHA -------------------
with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Resultados de la predicción")

    modelo = None
    pred_df = None

    # Cargar clases
    try:
        class_names = cargar_clases(CLASS_NAMES_PATH)
    except Exception as e:
        st.error(f"Error al cargar las clases: {e}")
        class_names = None

    # Botón de predicción
    if st.button("🔍 Clasificar ave", use_container_width=True):
        if imagen is None:
            st.warning("Primero sube una imagen para analizar.")
        elif class_names is None:
            st.warning("No se pudieron cargar las clases. Revisa el archivo class_names.txt.")
        else:
            try:
                with st.spinner(f"Cargando modelo {modelo_seleccionado} y realizando predicción…"):
                    # Cargar modelo desde caché
                    modelo = cargar_modelo(modelo_seleccionado)
                    input_size = MODEL_CONFIG[modelo_seleccionado]["input_size"]

                    # Preprocesar imagen
                    img_resized = imagen.resize(input_size)
                    img_array = np.array(img_resized) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    # Predicción
                    preds = modelo.predict(img_array)[0]  # vector 1D

                    # Ajustar longitud por si hay desajuste leve
                    n = min(len(preds), len(class_names))
                    preds = preds[:n]
                    clases = class_names[:n]

                    pred_df = pd.DataFrame(
                        {
                            "Especie": clases,
                            "Probabilidad": preds,
                        }
                    ).sort_values("Probabilidad", ascending=False)

                    # Mostrar resultado principal
                    top_row = pred_df.iloc[0]
                    st.success(
                        f"**Ave predicha:** {top_row['Especie']} "
                        f"con probabilidad {top_row['Probabilidad']*100:.2f}%"
                    )

                    st.write("### 🔢 Probabilidades por especie")
                    st.dataframe(
                        pred_df.style.format({"Probabilidad": "{:.4f}"}),
                        use_container_width=True,
                    )

                    st.write("### 📈 Top 5 clases (gráfico)")
                    top5 = pred_df.head(5).set_index("Especie")
                    st.bar_chart(top5)

            except Exception as e:
                st.error(f"Error al realizar la predicción: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
