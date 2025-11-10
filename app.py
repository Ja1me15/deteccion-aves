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
/* Fondo con franja superior verde y resto celeste */
.stApp {
    background: linear-gradient(
        180deg,
        #80ba26 0%,
        #80ba26 35%,
        #00abc8 35%,
        #b2f2f9 100%
    );
}
/* ================================
   COLOR DEL TEXTO EN EL CAPTION DE IMÁGENES
   ================================ */
div[data-testid="stImageCaption"] p,
.stImage figcaption,
.stImage p {
    color: #000000 !important;
    font-weight: 600;
    text-align: center;
}


/* Color del texto del caption de las imágenes */
.stImage > figcaption {
    color: #000000 !important; /* negro */
    font-weight: 600;
    text-align: center;
}

/* Ajuste del bloque del encabezado */
.header-container {
    margin-top: 2rem;       /* baja todo el bloque */
    margin-bottom: 1rem;    /* espacio con lo que sigue */
}

/* Título principal */
.header-container h1 {
    margin-top: 0.5rem;
    margin-bottom: 0.3rem;
    font-size: 2.4rem;      /* puedes ajustar el tamaño */
}

/* Subtítulo */
.header-container p {
    margin-top: 0;
    font-size: 1.05rem;
}


/* Contenedor principal */
.block-container {
    background-color: rgba(255, 255, 255, 0.0);
    padding: 1.5rem 1.5rem 3rem 1.5rem;
}

/* Tarjetas blancas */
.card {
    background-color: #FFFFFF;
    border-radius: 18px;
    padding: 1.8rem;
    box-shadow: 0 6px 14px rgba(0,0,0,0.1);
    margin-bottom: 1.2rem;
}

/* Títulos principales */
h1 {
    color: #FFFFFF !important;
    text-align: center;
    font-weight: 800;
}

/* Subtítulos dentro de tarjetas */
.card h2 {
    color: #000000 !important;
    margin-top: 0;
    font-weight: 700;
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

/* Textos de alertas (modelo cargado, ave predicha) */
.stAlert {
    color: #000000 !important;
    background-color: rgba(255,255,255,0.9);
    border-radius: 14px;
}

/* Selectbox y file uploader */
.stSelectbox > div > div,
.stFileUploader > div {
    border-radius: 999px;
}

/* Tabla de resultados */
.dataframe {
    color: #000000;
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

    model = tf.keras.models.load_model(
        model_path,
        safe_mode=False,
        compile=False,
    )
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
    <div class="header-container" style="text-align:center;">
        <h1>🦜 Detección de Aves</h1>
        <p>Proyecto con dos modelos de Deep Learning para clasificación de aves.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ===========================================================
# LAYOUT PRINCIPAL
# ===========================================================
col_left, col_right = st.columns([1.1, 1.1])

# ------------------------ COLUMNA IZQUIERDA -----------------
with col_left:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h2>⚙️ Configuración del modelo</h2>", unsafe_allow_html=True)

        modelo_seleccionado = st.selectbox(
            "Selecciona el modelo:",
            list(MODEL_CONFIG.keys()),
            index=0,
        )
        st.success(f"Modelo **{modelo_seleccionado}** cargado exitosamente.")

        st.markdown("---")
        st.markdown("<h2>📤 Cargar imagen</h2>", unsafe_allow_html=True)
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
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h2>📊 Resultados de la predicción</h2>", unsafe_allow_html=True)

        modelo = None
        pred_df = None

        try:
            class_names = cargar_clases(CLASS_NAMES_PATH)
        except Exception as e:
            st.error(f"Error al cargar las clases: {e}")
            class_names = None

        if st.button("🔍 Clasificar ave", use_container_width=True):
            if imagen is None:
                st.warning("Primero sube una imagen para analizar.")
            elif class_names is None:
                st.warning("No se pudieron cargar las clases. Revisa el archivo class_names.txt.")
            else:
                try:
                    with st.spinner(f"Clasificando con {modelo_seleccionado}..."):
                        modelo = cargar_modelo(modelo_seleccionado)
                        input_size = MODEL_CONFIG[modelo_seleccionado]["input_size"]

                        img_resized = imagen.resize(input_size)
                        img_array = np.array(img_resized) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)

                        preds = modelo.predict(img_array)[0]
                        n = min(len(preds), len(class_names))
                        preds = preds[:n]
                        clases = class_names[:n]

                        pred_df = pd.DataFrame({
                            "Especie": clases,
                            "Probabilidad": preds,
                        }).sort_values("Probabilidad", ascending=False)

                        top_row = pred_df.iloc[0]
                        st.success(
                            f"Ave predicha: **{top_row['Especie']}** con probabilidad **{top_row['Probabilidad']*100:.2f}%**"
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



