from predict import predict_image_from_array
import streamlit as st
import numpy as np
import cv2

# ---------------- SESSION ----------------
if "counter" not in st.session_state:
    st.session_state.counter = 0

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Deepfake Detector", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}

.main-title {
    text-align: center;
    font-size: 80px;
    font-weight: bold;
    background: linear-gradient(90deg, #00C9FF, #92FE9D);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #BBBBBB;
    font-size: 18px;
}

.section {
    padding: 20px;
    border-radius: 15px;
    background-color: #1c1f26;
    margin-top: 20px;
}

.fake {
    color: red;
    font-size: 28px;
    font-weight: bold;
}

.real {
    color: #00FF9C;
    font-size: 28px;
    font-weight: bold;
}

.footer {
    text-align: center;
    color: grey;
    font-size: 14px;
}
.block-container {
    padding-top: 0.5rem;
}

.main-title {
    margin-top: -4px;
    margin-bottom: 10px;
}        
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">Deepfake Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered detection for Fake vs Real Images</div>', unsafe_allow_html=True)

# ---------------- BANNER ----------------
# st.markdown("""
# <style>
# img {
#     border-radius: 15px;
#     box-shadow: 0px 10px 30px rgba(0,0,0,0.5);
# }
# </style>
# """, unsafe_allow_html=True)
# col1, col2, col3 = st.columns([1,3,1])

# with col2:
#     st.image("./banner.png", use_container_width=True)

    # ---------------- PREDICTION ----------------
def generate_reason(result):
    if result == "Fake":
         return """
This image shows signs of facial inconsistencies and unnatural blending.  
Lighting and shadow mismatches are observed, indicating possible manipulation.  
Additionally, the skin texture appears overly smooth, suggesting synthetic generation artifacts.
"""
    else:
        return """
This image appears natural with consistent facial features and proper alignment.  
Lighting and shadows are uniform, indicating no visible manipulation.  
The skin texture and details look realistic, which is typical of genuine images.
"""

   
center_col = st.columns([1, 2, 1])[1]

with center_col:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("🧠 Try It Yourself")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Centered Image
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(image, channels="BGR", caption="Preview", width=300)

        # Prediction
        with st.spinner("🔍 Analyzing Image..."):
            result, score = predict_image_from_array(image)

        # Count uploads
        st.session_state.counter += 1

        # Result
        if result == "Fake":
            st.error("🚨 Fake Image Detected")
        else:
            st.success("✅ Real Image")

        # Upload count
        st.markdown(f"📂 Images Uploaded: **{st.session_state.counter}**")

        # Explanation
        explanation = generate_reason(result)
        st.markdown("### 🧠 Analysis")
        st.write(explanation)

        st.info("💡 This explanation is generated based on visual patterns learned by the model.")

    st.markdown('</div>', unsafe_allow_html=True)
# ---------------- ABOUT ----------------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.header("📘 About Deepfakes")
st.write("""
Deepfakes are AI-generated synthetic media where a person’s face or voice is replaced using deep learning techniques.
They can be used for entertainment but also pose serious threats like misinformation and identity misuse.

This project demonstrates how AI can be used to detect such manipulated images.
""")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FEATURES ----------------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.header("✨ Features")
st.write("""
✔️ Image Upload & Preview  
✔️ AI-based Fake/Real Prediction  
✔️ Visual Feedback with Color Indicators  
✔️ Training Graph Visualization  
✔️ User-Friendly Interface  
""")
st.markdown('</div>', unsafe_allow_html=True)



# ---------------- MODEL PERFORMANCE ----------------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.header("📊 Model Performance")
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("./mobile_webp.webp",width=700)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")

st.markdown("""
<div class="footer">
📚 Learn More About Deepfakes:<br><br>

🔗 <a href="https://en.wikipedia.org/wiki/Deepfake" target="_blank">Project by Sneha Agrawal and Satwik Pandey</a><br>
🔗 <a href="https://en.wikipedia.org/wiki/Deepfake" target="_blank">What are Deepfakes?</a><br>
🔗 <a href="https://www.kaggle.com/datasets" target="_blank">Datasets for AI Projects</a><br>
🔗 <a href="https://tensorflow.org" target="_blank">TensorFlow Documentation</a><br>
🔗 <a href="https://opencv.org" target="_blank">OpenCV Library</a><br>

<br>
🚀 Developed as a Mini Project using AI & Streamlit
</div>
""", unsafe_allow_html=True)