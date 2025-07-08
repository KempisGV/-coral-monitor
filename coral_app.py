import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# Configurar los ajustes de la página
st.set_page_config(
    page_title="🪸 Monitor de Salud del Coral", # Título de la página
    page_icon="🪸", # Icono de la página
    layout="wide", # Diseño de página amplio
    initial_sidebar_state="expanded" # Estado inicial de la barra lateral
)

# CSS personalizado
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

# Inicializar el estado de la sesión
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []


@st.cache_resource
def load_model():
    """Cargar el modelo entrenado con caché""" # Docstring: Cargar el modelo entrenado con caché
    try:
        model = tf.keras.models.load_model('final_model_vgg16.h5')
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}") # Mensaje de error al cargar el modelo
        st.info(
            "Por favor, asegúrese de que 'final_model_vgg16.h5' esté en el mismo directorio que este script.") # Información sobre la ubicación del modelo
        return None


def preprocess_image(image, target_size=(224, 224)):
    """Preprocesar la imagen para la predicción del modelo con manejo de errores""" # Docstring: Preprocesar imagen
    try:
        # Convertir imagen PIL a array numpy
        img_array = np.array(image)

        # Depuración: Imprimir información de la imagen
        st.write(f"🔍 Depuración: Forma de la imagen original: {img_array.shape}") # Mensaje de depuración: forma original
        st.write(f"🔍 Depuración: Modo de imagen: {image.mode}") # Mensaje de depuración: modo de imagen

        # Manejar diferentes modos y canales de imagen
        if len(img_array.shape) == 2:  # Escala de grises (1 canal)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            st.info("📝 Se convirtió la imagen en escala de grises a RGB") # Información: conversión a RGB

        elif len(img_array.shape) == 3:
            if img_array.shape[2] == 4:  # RGBA (4 canales) - ESTE ES SU PROBLEMA
                # Método 1: Convertir RGBA a RGB eliminando el canal alfa
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                st.info("📝 Se convirtió la imagen RGBA a RGB (se eliminó el canal alfa)") # Información: conversión RGBA a RGB

            elif img_array.shape[2] == 3:  # RGB (3 canales) - ¡Perfecto!
                pass  # Ya está en el formato correcto

            elif img_array.shape[2] == 1:  # Canal único en un array 3D
                img_array = np.repeat(img_array, 3, axis=2)
                st.info("📝 Se convirtió el canal único a RGB") # Información: conversión de canal único a RGB

            else:
                st.error(f"❌ Formato de imagen no compatible: {img_array.shape[2]} canales") # Error: formato no compatible
                return None

        # Verificación final de la forma
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            st.error(f"❌ Falló el preprocesamiento de la imagen. Forma final: {img_array.shape}") # Error: preprocesamiento fallido
            return None

        st.success(f"✅ Imagen preprocesada exitosamente. Forma: {img_array.shape}") # Éxito: imagen preprocesada

        # Redimensionar imagen
        img_resized = cv2.resize(img_array, target_size)

        # Normalizar los valores de píxel a [0, 1]
        img_normalized = img_resized.astype('float32') / 255.0

        # Añadir dimensión de lote
        img_batch = np.expand_dims(img_normalized, axis=0)

        st.write(f"🔍 Forma final del tensor: {img_batch.shape}") # Mensaje de depuración: forma final del tensor

        return img_batch

    except Exception as e:
        st.error(f"❌ Error al preprocesar la imagen: {str(e)}") # Error: al preprocesar imagen
        st.error(f"🐛 Forma de la imagen: {getattr(image, 'size', 'desconocida')}") # Error: forma de la imagen (depuración)
        st.error(f"🐛 Modo de imagen: {getattr(image, 'mode', 'desconocido')}") # Error: modo de imagen (depuración)
        return None


def create_confidence_chart(healthy_prob, bleached_prob):
    """Crear gráfico de confianza""" # Docstring: Crear gráfico de confianza
    fig = go.Figure(data=[
        go.Bar(
            x=['Saludable', 'Blanqueado'], # Etiqueta del eje X
            y=[healthy_prob, bleached_prob],
            marker_color=['#38ef7d', '#fc466b'],
            text=[f'{healthy_prob:.1f}%', f'{bleached_prob:.1f}%'],
            textposition='auto',
            textfont=dict(size=16, color='white'),
        )
    ])

    fig.update_layout(
        title={
            'text': 'Confianza de la Predicción', # Título del gráfico
            'x': 0.5,
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        xaxis_title="Estado del Coral", # Título del eje X
        yaxis_title="Confianza (%)", # Título del eje Y
        yaxis=dict(range=[0, 100]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        height=400
    )

    return fig


def create_history_chart():
    """Crear gráfico del historial de predicciones""" # Docstring: Crear gráfico del historial
    if not st.session_state.prediction_history:
        return None

    df = pd.DataFrame(st.session_state.prediction_history)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['healthy_confidence'],
        mode='lines+markers',
        name='Confianza Saludable', # Nombre de la línea
        line=dict(color='#38ef7d', width=3),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['bleached_confidence'],
        mode='lines+markers',
        name='Confianza Blanqueado', # Nombre de la línea
        line=dict(color='#fc466b', width=3),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title={
            'text': 'Historial de Predicciones', # Título del gráfico
            'x': 0.5,
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        xaxis_title="Tiempo", # Título del eje X
        yaxis_title="Confianza (%)", # Título del eje Y
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        height=400,
        hovermode='x unified'
    )

    return fig


def main():
    # Encabezado
    st.markdown('<h1 class="main-header">🪸 Monitor de Salud del Coral</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Evaluación de la Salud de los Arrecifes de Coral con IA</p>',
                unsafe_allow_html=True)

    # Barra lateral
    with st.sidebar:
        st.markdown("## 🎛️ Panel de Control") # Título del panel de control

        # Información del modelo
        st.markdown("""
        <div class="sidebar-content">
            <h3>🧠 Información del Modelo</h3>
            <p>Este modelo de IA analiza imágenes de coral para detectar signos de blanqueamiento y evaluar la salud general del coral.</p>
        </div>
        """, unsafe_allow_html=True)

        # Ajustes
        st.markdown("### ⚙️ Ajustes") # Título de ajustes
        show_confidence = st.checkbox("Mostrar Puntuaciones de Confianza", value=True) # Checkbox: mostrar confianza
        show_history = st.checkbox("Mostrar Historial de Predicciones", value=True) # Checkbox: mostrar historial
        auto_analyze = st.checkbox("Analizar automáticamente al cargar", value=True) # Checkbox: auto-analizar

        # Botón para borrar historial
        if st.button("🗑️ Borrar Historial", use_container_width=True): # Botón: borrar historial
            st.session_state.prediction_history = []
            st.success("¡Historial borrado!") # Mensaje de éxito: historial borrado

    # Cargar modelo
    model = load_model()

    if model is None:
        st.stop()

    # Área de contenido principal
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📸 Cargar Imagen de Coral") # Título: cargar imagen de coral

        # Cargador de archivos con estilo personalizado
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Elija una imagen...", # Etiqueta del cargador de archivos
            type=['png', 'jpg', 'jpeg'],
            help="Cargue una imagen clara de coral para su análisis" # Texto de ayuda
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            # Mostrar imagen cargada
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen Cargada", use_container_width=True) # Título de la imagen cargada

            # Botón de análisis
            analyze_button = st.button("🔍 Analizar Salud del Coral", # Etiqueta del botón de análisis
                                       use_container_width=True, type="primary")

            if analyze_button or auto_analyze:
                with st.spinner("🧠 Analizando la salud del coral..."): # Mensaje del spinner
                    # Preprocesar imagen
                    processed_image = preprocess_image(image)

                    # Realizar predicción
                    prediction = model.predict(processed_image, verbose=0)

                    # Asumiendo clasificación binaria: [saludable, blanqueado]
                    healthy_prob = float(prediction[0][0]) * 100
                    bleached_prob = (1 - float(prediction[0][0])) * 100

                    # Determinar predicción
                    is_healthy = healthy_prob > bleached_prob
                    confidence = max(healthy_prob, bleached_prob)

                    # Almacenar en el historial
                    st.session_state.prediction_history.append({
                        'timestamp': datetime.now(),
                        'healthy_confidence': healthy_prob,
                        'bleached_confidence': bleached_prob,
                        'prediction': 'Saludable' if is_healthy else 'Blanqueado' # Resultado de la predicción
                    })

                    # Almacenar resultados en el estado de la sesión para mostrar en col2
                    st.session_state.current_prediction = {
                        'is_healthy': is_healthy,
                        'healthy_prob': healthy_prob,
                        'bleached_prob': bleached_prob,
                        'confidence': confidence
                    }

    with col2:
        st.markdown("### 📊 Resultados del Análisis") # Título de los resultados

        if hasattr(st.session_state, 'current_prediction'):
            pred = st.session_state.current_prediction

            # Resultado principal de la predicción
            if pred['is_healthy']:
                st.markdown(f"""
                <div class="healthy-card">
                    <h2>✅ Coral Saludable</h2> # Título: coral saludable
                    <h3>{pred['confidence']:.1f}% Confianza</h3> # Confianza
                    <p>¡El coral parece estar en buena salud!</p> # Mensaje: buena salud
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bleached-card">
                    <h2>⚠️ Coral Blanqueado</h2> # Título: coral blanqueado
                    <h3>{pred['confidence']:.1f}% Confianza</h3> # Confianza
                    <p>El coral muestra signos de blanqueamiento.</p> # Mensaje: signos de blanqueamiento
                </div>
                """, unsafe_allow_html=True)

            # Gráfico de confianza
            if show_confidence:
                fig = create_confidence_chart(pred['healthy_prob'],
                                              pred['bleached_prob'])
                st.plotly_chart(fig, use_container_width=True)

            # Métricas detalladas
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric(
                    label="🟢 Probabilidad Saludable", # Etiqueta: probabilidad saludable
                    value=f"{pred['healthy_prob']:.1f}%",
                    delta=f"{pred['healthy_prob'] - 50:.1f}%" if pred[
                                                                     'healthy_prob'] != 50 else None
                )

            with col2_2:
                st.metric(
                    label="🔴 Probabilidad Blanqueado", # Etiqueta: probabilidad blanqueado
                    value=f"{pred['bleached_prob']:.1f}%",
                    delta=f"{pred['bleached_prob'] - 50:.1f}%" if pred[
                                                                      'bleached_prob'] != 50 else None
                )

        else:
            st.info("👆 ¡Cargue una imagen para ver los resultados del análisis aquí!") # Mensaje: cargar imagen

    # Sección de historial
    if show_history and st.session_state.prediction_history:
        st.markdown("---")
        st.markdown("### 📈 Historial de Predicciones") # Título: historial de predicciones

        fig_history = create_history_chart()
        if fig_history:
            st.plotly_chart(fig_history, use_container_width=True)

        # Tabla de historial
        with st.expander("📋 Historial Detallado"): # Título del expander
            df_history = pd.DataFrame(st.session_state.prediction_history)
            df_history['timestamp'] = df_history['timestamp'].dt.strftime(
                '%Y-%m-%d %H:%M:%S')
            st.dataframe(df_history, use_container_width=True)

    # Pie de página
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🌊 Protegiendo arrecifes de coral con IA</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
