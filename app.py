"""
Streamlit Web Application for Signature Forgery Detector.
Provides an interactive web interface to verify handwriting signature authenticity using PyTorch Siamese Networks.
"""

import os
import sys
from PIL import Image
import torch
import torch.nn.functional as F
import streamlit as st

# Append root directory to python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils import preprocess_image
from src.model import SiameseNet, EmbeddingNet

# Streamlit Page Setup
st.set_page_config(
    page_title="Signature Forgery Detector",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #818CF8, #C084FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }

    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 1rem;
    }
    .match-card {
        background-color: #F0FDF4;
        border: 2px solid #22C55E;
        color: #15803D;
    }
    .nomatch-card {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        color: #B91C1C;
    }
    .badge {
        font-size: 1.8rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_siamese_model(model_path: str = "models/siamese_model.pt") -> SiameseNet:
    """
    Loads and caches the trained Siamese Neural Network model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNet(EmbeddingNet()).to(device)

    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            return model
        except Exception as e:
            st.warning(f"Could not load custom model checkpoint ({e}). Using initialized network weights.")
    
    model.eval()
    return model


def main():
    st.markdown('<div class="main-title">✍️ Signature Forgery Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Deep Learning Signature Verification using Siamese Networks & Contrastive Loss</div>', unsafe_allow_html=True)

    # Sidebar Information
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        threshold = st.slider("Distance Threshold", min_value=0.2, max_value=1.8, value=0.85, step=0.05,
                              help="Pairs with distance below this threshold are classified as genuine matches.")

        st.divider()
        st.header("📌 Project Details")
        st.markdown("""
        **Architecture**: Siamese CNN (Shared Weights)  
        **Embedding Dimension**: 128-d Vector  
        **Loss Function**: Contrastive Loss  
        **Benchmark Dataset**: CEDAR Signature Dataset  
        """)

        st.divider()
        st.markdown("[🔗 GitHub Repository](https://github.com/example/signature-forgery-detector)")
        st.caption("⚠️ **Educational Disclaimer**: This system is designed for portfolio and academic purposes only. It is not a substitute for legal forensic document analysis.")

    # Model Loading
    model_path = "models/siamese_model.pt"
    if not os.path.exists(model_path):
        st.info("💡 *Note: Trained model checkpoint 'models/siamese_model.pt' was not found. Using initialized network for visual demo. Run `python src/train.py` to train your own weights.*")

    model = load_siamese_model(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image Upload Section
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Reference Signature (Genuine)")
        ref_file = st.file_uploader("Upload reference genuine signature", type=["png", "jpg", "jpeg"], key="ref")
        if ref_file:
            ref_image = Image.open(ref_file).convert("RGB")
            st.image(ref_image, caption="Reference Signature", use_container_width=True)

    with col2:
        st.subheader("2. Signature to Verify")
        test_file = st.file_uploader("Upload signature for verification", type=["png", "jpg", "jpeg"], key="test")
        if test_file:
            test_image = Image.open(test_file).convert("RGB")
            st.image(test_image, caption="Target Signature", use_container_width=True)

    # Verification Trigger
    st.divider()
    if st.button("🔍 Verify Signatures", type="primary", use_container_width=True):
        if not ref_file or not test_file:
            st.error("Please upload BOTH signature images before running verification.")
            return

        with st.spinner("Analyzing stroke patterns & computing embedding distance..."):
            # Preprocess images
            tensor1 = preprocess_image(ref_image).to(device)
            tensor2 = preprocess_image(test_image).to(device)

            # Compute embeddings
            with torch.no_grad():
                emb1, emb2 = model(tensor1, tensor2)
                distance = F.pairwise_distance(emb1, emb2).item()

            is_match = distance <= threshold

        # Display Results
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if is_match:
                st.markdown(f"""
                    <div class="result-card match-card">
                        <div class="badge">✅ MATCH VERIFIED</div>
                        <p>The uploaded signatures belong to the <b>same person</b>.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-card nomatch-card">
                        <div class="badge">❌ FORGERY / MISMATCH DETECTED</div>
                        <p>The signatures exhibit distinct stroke characteristics (<b>different person or forged</b>).</p>
                    </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.metric(label="Embedding Euclidean Distance", value=f"{distance:.4f}", delta=f"Threshold: {threshold:.2f}", delta_color="inverse")
            st.progress(min(1.0, distance / 2.0))
            st.caption(f"Score < {threshold:.2f} $\\rightarrow$ Match | Score $\\ge$ {threshold:.2f} $\\rightarrow$ Forgery/Mismatch")


if __name__ == "__main__":
    main()
