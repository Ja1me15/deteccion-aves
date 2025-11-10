import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# ===========================================================
# CONFIGURACIÓN GENERAL
# ===========================================================
st.set_page_config(
    page_title="Detección de Aves 🦜",
    page_icon="🦜",
    layout="wide",
)

# ===========================================================
# ESTILOS PERSONALIZADOS
# ===========================================================
st.markdown(
    """
<style>
/* Fondo: verde arriba, azul abajo */
.stApp {
    background: linear-gradient(
        180deg,
        #80ba26 0%,
        #80ba26 35%,
        #00abc8 35%,
        #b2f2f9 100%
    );
}

/* Contenedor principal */
.block-container {
    background-color: rgba(255, 255, 255, 0.0);
    padding: 1.5rem 1.5rem 3rem 1.5rem;
}

/* CABECERA GENERAL */
.header-container {
    text-align: center;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
}

.header-container h1 {
    margin: 0.5rem 0 0.2rem 0;
    font-size: 2.6rem;
    color: #FFFFFF;
    font-weight: 900;
}

.header-container p {
    margin: 0;
    font-size: 1.05rem;
    color: #FFFFFF;
}

/* Títulos de secciones (Configuración / Resultados) */
.section-title {
    font-size: 1.8rem;
    font-weight: 800;
    color: #000000;
    display: flex;
    align-items: center;
    margin: 1.2rem 0 1rem 0;
}

.section-title span {
    margin-right: 10px;
}

/* Botones */
.stButton>button {
    background-color: #80ba26;
    color: #FFFFFF;
    font-weight: 700;
    border-radius: 999px;
    padding: 0.6rem 1.4rem;
    border: none;
}
.stButton>button:hover {
    background-color: #6ca01e;
    color: #FFFFFF;
}

/* Selectbox y file uploader */
.stSelectbox > div > div,
.stFileUploader > div {
    border-radius: 999px;
}

/* Alertas (modelo cargado, ave predicha) con texto negro */
.stAlert {
    border-radius: 14px;
    color: #000000 !important;
    background-color: rgba(255,255,255,0.9);
}

/* Caption de imagen (Imagen cargada correctamente) */
div[data-testid="stImageCaption"] p,
.stImage figcaption,
.stImage p {
    color: #000000 !important;
    font-weight: 600;
    text-align: center;
}

/* Compactar espacio entre alerta y “Cargar imagen” */
div[data-testid="stVerticalBlock"] > div:has(> .stAlert) {
    margin-bottom: 0.6rem !important;
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
    },
    "VGG16": {
        "path": "modelos/vgg16.keras",
        "input_size": (224, 224),
    },
}

CLASS_NAMES_PATH = "class_names.txt"


@st.cache_resource(show_spinner="Cargando modelo seleccionado…")
def cargar_modelo(nombre_modelo: str):
    config = MODEL_CONFIG.get(nombre_modelo)
    if config is None:
        raise ValueError(f"No existe configuración para el modelo: {nombre_modelo}")

    model_path = config["path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

    model = tf.keras.models.load_model(model_path, safe_mode=False, compile=False)
    return model


@st.cache_data
def cargar_clases(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de clases en: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f.readlines() if l.strip()]


# ===========================================================
# CABECERA
# ===========================================================
st.markdown(
    """
    <div class="header-container">
        <h1>🦜 Detección de Aves</h1>
        <p>Aplicación de inteligencia artificial para el reconocimiento y estudio de aves representativas del Tolima mediante modelos de Deep Learning..</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===========================================================
# LAYOUT PRINCIPAL
# ===========================================================
col_left, col_right = st.columns([1, 1])

# ------------------------ COLUMNA IZQUIERDA -----------------
with col_left:
    st.markdown(
        '<div class="section-title"><span>⚙️</span>Configuración del modelo</div>',
        unsafe_allow_html=True,
    )

    modelo_seleccionado = st.selectbox(
        "Selecciona el modelo:",
        list(MODEL_CONFIG.keys()),
        index=0,
    )

    try:
        modelo = cargar_modelo(modelo_seleccionado)
        st.success(f"Modelo **{modelo_seleccionado}** cargado exitosamente.")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        modelo = None
    st.subheader("📤 Cargar imagen")
    archivo_imagen = st.file_uploader(
        "Sube una imagen de un ave (JPG o PNG):",
        type=["jpg", "jpeg", "png"],
    )

    imagen = None
    if archivo_imagen is not None:
        imagen = Image.open(archivo_imagen).convert("RGB")
        st.image(imagen, caption="Imagen cargada correctamente ✅", use_column_width=True)

# ------------------------ COLUMNA DERECHA -------------------
with col_right:
    st.markdown(
        '<div class="section-title"><span>📊</span>Resultados de la predicción</div>',
        unsafe_allow_html=True,
    )

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
            st.warning("No se pudieron cargar las clases. Revisa el archivo class_names.txt.")
        else:
            try:
                with st.spinner(f"Clasificando con {modelo_seleccionado}..."):
                    input_size = MODEL_CONFIG[modelo_seleccionado]["input_size"]

                    img_resized = imagen.resize(input_size)
                    img_array = np.array(img_resized) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    preds = modelo.predict(img_array)[0]

                    n = min(len(preds), len(class_names))
                    preds = preds[:n]
                    clases = class_names[:n]

                    pred_df = pd.DataFrame(
                        {"Especie": clases, "Probabilidad": preds}
                    ).sort_values("Probabilidad", ascending=False)

                    top_row = pred_df.iloc[0]
                    st.success(
                        f"Ave predicha: **{top_row['Especie']}** "
                        f"con probabilidad **{top_row['Probabilidad']*100:.2f}%**"
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






