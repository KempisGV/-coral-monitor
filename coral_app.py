import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# ======== CONFIGURACIÓN DE DESCARGA DEL MODELO DESDE GOOGLE DRIVE ========
MODEL_DIR = "/data" if os.path.isdir("/data") else "."
MODEL_PATH = os.path.join(MODEL_DIR, "final_model_vgg16.h5")
GDRIVE_ID = "1qpTNX_vE_J4WLOp-BcqtlbintdqsK8-8"  # <-- ID de tu archivo en Google Drive

if not os.path.exists(MODEL_PATH):
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
        st.warning("Descargando modelo desde Google Drive. Esto puede tardar unos minutos la primera vez...")
        gdown.download(url, MODEL_PATH, quiet=False)
        st.success("¡Modelo descargado exitosamente!")
    except Exception as e:
        st.error(f"No se pudo descargar el modelo: {str(e)}")
        st.stop()
else:
    print(f"Modelo ya existe en {MODEL_PATH}")

# ======== CONFIGURACIÓN DE LA PÁGINA ========
st.set_page_config(
    page_title="🪸 Monitor de Salud del Coral",
    page_icon="🪸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======== CSS personalizado ========
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .healthy-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .bleached-card {
        background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .upload-section {
        border: 3px dashed #4ECDC4;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        background: rgba(78, 205, 196, 0.05);
        margin: 2rem 0;
    }
    .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .prediction-box {
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ======== ESTADO DE SESIÓN ========
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

@st.cache_resource
def load_model():
    """Cargar el modelo entrenado con caché"""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        st.info(
            f"Por favor, asegúrese de que 'final_model_vgg16.h5' esté en {MODEL_PATH} o sea accesible.")
        return None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocesar la imagen para la predicción del modelo con manejo de errores"""
    try:
        img_array = np.array(image)
        st.write(f"🔍 Depuración: Forma de la imagen original: {img_array.shape}")
        st.write(f"🔍 Depuración: Modo de imagen: {image.mode}")
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            st.info("📝 Se convirtió la imagen en escala de grises a RGB")
        elif len(img_array.shape) == 3:
            if img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                st.info("📝 Se convirtió la imagen RGBA a RGB (se eliminó el canal alfa)")
            elif img_array.shape[2] == 3:
                pass
            elif img_array.shape[2] == 1:
                img_array = np.repeat(img_array, 3, axis=2)
                st.info("📝 Se convirtió el canal único a RGB")
            else:
                st.error(f"❌ Formato de imagen no compatible: {img_array.shape[2]} canales")
                return None
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            st.error(f"❌ Falló el preprocesamiento de la imagen. Forma final: {img_array.shape}")
            return None
        st.success(f"✅ Imagen preprocesada exitosamente. Forma: {img_array.shape}")
        img_resized = cv2.resize(img_array, target_size)
        img_normalized = img_resized.astype('float32') / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        st.write(f"🔍 Forma final del tensor: {img_batch.shape}")
        return img_batch
    except Exception as e:
        st.error(f"❌ Error al preprocesar la imagen: {str(e)}")
        st.error(f"🐛 Forma de la imagen: {getattr(image, 'size', 'desconocida')}")
        st.error(f"🐛 Modo de imagen: {getattr(image, 'mode', 'desconocido')}")
        return None

def create_confidence_chart(healthy_prob, bleached_prob):
    fig = go.Figure(data=[
        go.Bar(
            x=['Saludable', 'Blanqueado'],
            y=[healthy_prob, bleached_prob],
            marker_color=['#38ef7d', '#fc466b'],
            text=[f'{healthy_prob:.1f}%', f'{bleached_prob:.1f}%'],
            textposition='auto',
            textfont=dict(size=16, color='white'),
        )
    ])
    fig.update_layout(
        title={
            'text': 'Confianza de la Predicción',
            'x': 0.5,
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        xaxis_title="Estado del Coral",
        yaxis_title="Confianza (%)",
        yaxis=dict(range=[0, 100]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        height=400
    )
    return fig

def create_history_chart():
    if not st.session_state.prediction_history:
        return None
    df = pd.DataFrame(st.session_state.prediction_history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['healthy_confidence'],
        mode='lines+markers',
        name='Confianza Saludable',
        line=dict(color='#38ef7d', width=3),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['bleached_confidence'],
        mode='lines+markers',
        name='Confianza Blanqueado',
        line=dict(color='#fc466b', width=3),
        marker=dict(size=8)
    ))
    fig.update_layout(
        title={
            'text': 'Historial de Predicciones',
            'x': 0.5,
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        xaxis_title="Tiempo",
        yaxis_title="Confianza (%)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        height=400,
        hovermode='x unified'
    )
    return fig

def main():
    st.markdown('<h1 class="main-header">🪸 Monitor de Salud del Coral</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Evaluación de la Salud de los Arrecifes de Coral con IA</p>', unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("## 🎛️ Panel de Control")
        st.markdown("""
        <div class="sidebar-content">
            <h3>🧠 Información del Modelo</h3>
            <p>Este modelo de IA analiza imágenes de coral para detectar signos de blanqueamiento y evaluar la salud general del coral.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("### ⚙️ Ajustes")
        show_confidence = st.checkbox("Mostrar Puntuaciones de Confianza", value=True)
        show_history = st.checkbox("Mostrar Historial de Predicciones", value=True)
        auto_analyze = st.checkbox("Analizar automáticamente al cargar", value=True)
        if st.button("🗑️ Borrar Historial", use_container_width=True):
            st.session_state.prediction_history = []
            st.success("¡Historial borrado!")
    model = load_model()
    if model is None:
        st.stop()
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 📸 Cargar Imagen de Coral")
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Elija una imagen...",
            type=['png', 'jpg', 'jpeg'],
            help="Cargue una imagen clara de coral para su análisis"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen Cargada", use_container_width=True)
            analyze_button = st.button("🔍 Analizar Salud del Coral", use_container_width=True, type="primary")
            if analyze_button or auto_analyze:
                with st.spinner("🧠 Analizando la salud del coral..."):
                    processed_image = preprocess_image(image)
                    prediction = model.predict(processed_image, verbose=0)
                    healthy_prob = float(prediction[0][0]) * 100
                    bleached_prob = (1 - float(prediction[0][0])) * 100
                    is_healthy = healthy_prob > bleached_prob
                    confidence = max(healthy_prob, bleached_prob)
                    st.session_state.prediction_history.append({
                        'timestamp': datetime.now(),
                        'healthy_confidence': healthy_prob,
                        'bleached_confidence': bleached_prob,
                        'prediction': 'Saludable' if is_healthy else 'Blanqueado'
                    })
                    st.session_state.current_prediction = {
                        'is_healthy': is_healthy,
                        'healthy_prob': healthy_prob,
                        'bleached_prob': bleached_prob,
                        'confidence': confidence
                    }
    with col2:
        st.markdown("### 📊 Resultados del Análisis")
        if hasattr(st.session_state, 'current_prediction'):
            pred = st.session_state.current_prediction
            if pred['is_healthy']:
                st.markdown(f"""
                <div class="healthy-card">
                    <h2>✅ Coral Saludable</h2>
                    <h3>{pred['confidence']:.1f}% Confianza</h3>
                    <p>¡El coral parece estar en buena salud!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bleached-card">
                    <h2>⚠️ Coral Blanqueado</h2>
                    <h3>{pred['confidence']:.1f}% Confianza</h3>
                    <p>El coral muestra signos de blanqueamiento.</p>
                </div>
                """, unsafe_allow_html=True)
            if show_confidence:
                fig = create_confidence_chart(pred['healthy_prob'], pred['bleached_prob'])
                st.plotly_chart(fig, use_container_width=True)
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric(
                    label="🟢 Probabilidad Saludable",
                    value=f"{pred['healthy_prob']:.1f}%",
                    delta=f"{pred['healthy_prob'] - 50:.1f}%" if pred['healthy_prob'] != 50 else None
                )
            with col2_2:
                st.metric(
                    label="🔴 Probabilidad Blanqueado",
                    value=f"{pred['bleached_prob']:.1f}%",
                    delta=f"{pred['bleached_prob'] - 50:.1f}%" if pred['bleached_prob'] != 50 else None
                )
        else:
            st.info("👆 ¡Cargue una imagen para ver los resultados del análisis aquí!")
    if show_history and st.session_state.prediction_history:
        st.markdown("---")
        st.markdown("### 📈 Historial de Predicciones")
        fig_history = create_history_chart()
        if fig_history:
            st.plotly_chart(fig_history, use_container_width=True)
        with st.expander("📋 Historial Detallado"):
            df_history = pd.DataFrame(st.session_state.prediction_history)
            df_history['timestamp'] = df_history['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(df_history, use_container_width=True)
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🌊 Protegiendo arrecifes de coral con IA</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
