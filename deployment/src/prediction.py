import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os

@st.cache_resource
def load_drone_model():
    """Builds Xception architecture and loads weights with fallback to full load."""
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'model.keras') 
    
    if not os.path.exists(model_path):
        return None

    try:
        from tensorflow.keras.applications import Xception
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dense, Dropout

        # Rebuild architecture
        base_model = Xception(weights=None, include_top=False, input_shape=(150, 150, 3))
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dense(256, activation='relu'),
            Dropout(0.4),
            Dense(6, activation='softmax')
        ])
        model.load_weights(model_path)
        return model
    except Exception:
        try:
            return tf.keras.models.load_model(model_path, compile=False)
        except:
            return None

def menu1():
    """Main Prediction Menu - Drone Intelligence with Static Anchoring."""
    st.markdown('<h1 class="matrix-header">🚁 Drone Intelligence</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: gray;">Adaptive Terrain Classification System</p>', unsafe_allow_html=True)
    st.divider()

    model = load_drone_model()
    if model is None:
        st.error("Model Error: Failed to link neural engine.")
        return

    class_names = ['Buildings', 'Forest', 'Glacier', 'Mountain', 'Sea', 'Street']
    SAFETY_THRESHOLD = 0.69 

    # FIXED INPUT SECTION
    with st.container():
        st.subheader("📥 Data Ingestion")
        up = st.file_uploader("Upload Drone Vision...", type=['jpg','png','jpeg'], key="main_pred_u")
        st.caption(f"Security Protocol: Objects with confidence < {SAFETY_THRESHOLD} will be flagged as UNKNOWN.")
    
    st.divider()

    # STATIC ANCHORS

    status_msg = st.empty()
    image_viewport = st.empty()
    analysis_header = st.empty()
    result_container = st.container()

    if up:
        try:
            # Pre-Process & Display Image Immediately
            file_bytes = np.frombuffer(up.getvalue(), np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            image_viewport.image(img_rgb, caption=f"Uplink Source: {up.name}", use_container_width=True)
            status_msg.warning("🧠 Analyzing Neural Pathway... Standby.")

            # Neural Inference (The Heavy Lifting)
            res = cv2.resize(img_rgb, (150, 150))
            arr = np.expand_dims(res.astype('float32') / 255.0, axis=0)
            preds = model.predict(arr, verbose=0)
            score = np.max(preds)
            idx = np.argmax(preds)

            # Final Output Injection
            status_msg.empty() # Hapus pesan loading
            analysis_header.subheader("🔍 Analysis Result")
            
            with result_container:
                if score < SAFETY_THRESHOLD:
                    st.error("🚨 **UNKNOWN OBJECT / OOD DETECTED**")
                    st.info(f"Target confidence level ({score:.4f}) below safety threshold of {SAFETY_THRESHOLD}.")
                    st.metric("Confidence Score", f"{score:.4f}", delta="OUT OF BOUNDS", delta_color="inverse")
                else:
                    st.success(f"✅ **CLASSIFICATION RESULT: {class_names[idx].upper()}**")
                    c1, c2 = st.columns(2)
                    c1.metric("Target Class", class_names[idx])
                    c2.metric("Confidence", f"{score:.4f}")
                
        except Exception as e:
            status_msg.error(f"Processing Error: {e}")
    else:
        # State standby jika belum ada upload
        status_msg.info("💡 Waiting for drone telemetry. Please upload an image.")