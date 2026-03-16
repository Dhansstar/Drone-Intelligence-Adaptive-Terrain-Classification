import streamlit as st
import cv2
import matplotlib.pyplot as plt
import numpy as np

def run_analysis():
    st.markdown('<h1 class="matrix-header">GEOSPATIAL EDA ANALYSIS</h1>', unsafe_allow_html=True)
    
    # INPUT SECTION
    up = st.file_uploader("Upload for Diagnostics...", type=['jpg','png','jpeg'], key="eda_u")
    st.divider()

    if up:
        with st.container():
            try:
                # Progress indicator
                status = st.status("📡 Establishing Uplink & Scanning Terrain...", expanded=True)
                
                # Processing Data
                file_bytes = np.frombuffer(up.getvalue(), np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # ORIGINAL IMAGE DISPLAY
                # Menampilkan gambar asli yang lo upload (img2.jpg)
                st.subheader("🖼️ Original Drone Viewport")
                st.image(img_rgb, use_container_width=True, caption=f"Source: {up.name} ({up.size/1024:.1f} KB)")
                st.divider()

                # SPECTRAL SCAN
                st.subheader("📊 Spectral Histogram")
                
                fig, ax = plt.subplots(figsize=(10, 5)) 
                fig.patch.set_facecolor('#0c0c0c')
                ax.set_facecolor('#0c0c0c')
                
                r_avg = np.mean(img_rgb[:,:,0])
                g_avg = np.mean(img_rgb[:,:,1])
                b_avg = np.mean(img_rgb[:,:,2])

                for i, col in enumerate(['red', 'green', 'blue']):
                    h = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
                    ax.plot(h, color=col, linewidth=2.5)
                
                ax.tick_params(colors='#00FF00', labelsize=10)
                ax.grid(color='#004400', linestyle='--', alpha=0.3)
                
                st.pyplot(fig)
                plt.close(fig)

                # Insights
                if g_avg > r_avg and g_avg > b_avg:
                    st.info("🌿 **Vegetation Bias:** Strong Green channel detected. High probability of Forest canopy.")
                elif b_avg > r_avg and b_avg > g_avg:
                    st.info("🌊 **Hydrological Bias:** High Blue intensity. Likely Water bodies or Glacial ice.")
                else:
                    st.info("🏢 **Neutral/Urban Bias:** Balanced RGB spectrum. Typical of concrete or asphalt structures.")

                st.divider()

                # STRUCTURAL SCAN 
                st.subheader("🔍 Structural Canny Edge Scan")
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                edge = cv2.Canny(gray, 100, 200)
                
                st.image(edge, use_container_width=True, caption="Structural Edge Mapping (Texture)")

                edge_density = np.sum(edge/255) / (edge.shape[0] * edge.shape[1])
                
                if edge_density > 0.08:
                    st.success("🏙️ **Structural Insight:** High complexity detected. Consistent with Urban Man-made structures.")
                else:
                    st.success("⛰️ **Structural Insight:** Low to Medium complexity. Consistent with Natural Terrain.")

                # Finalizing
                status.update(label="✅ Analysis Complete", state="complete", expanded=False)

                # Diagnostic Summary
                st.code(f"Filename: {up.name} | Density: {edge_density:.4f} | RGB Mean: {r_avg:.1f}, {g_avg:.1f}, {b_avg:.1f}")

            except Exception as e:
                st.error(f"System Crash: {e}")
    else:
        st.info("💡 Diagnostic system offline. Please upload geospatial imagery to initiate scan.")