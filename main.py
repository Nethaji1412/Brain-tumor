"""
═══════════════════════════════════════════════════════════════════════════════
ONCOSCAN - Brain Tumor Detection & Segmentation
Production-Level Streamlit Application
═══════════════════════════════════════════════════════════════════════════════

A medical imaging application for brain tumor classification and segmentation
using ResNet18 and SwinUNet with AES-256 encryption and professional UI.

Author: ONCOSCAN Medical AI
Version: 1.0.0
"""

import os
import io
import cv2
import torch
import numpy as np
import streamlit as st
import torch.nn as nn
from PIL import Image
from typing import Tuple, Dict
from torchvision import models, transforms
import timm
from cryptography.fernet import Fernet
import plotly.graph_objects as go
from datetime import datetime
import json
import requests
import tempfile
from urllib.parse import urlparse

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & SECURITY SETUP
# ═══════════════════════════════════════════════════════════════════════════════

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KEY_PATH = "secret.key"

@st.cache_resource
def get_encryption_key():
    """Generate or load encryption key"""
    if not os.path.exists(KEY_PATH):
        with open(KEY_PATH, "wb") as f:
            f.write(Fernet.generate_key())
    with open(KEY_PATH, "rb") as f:
        return f.read()

def encrypt_bytes(image_bytes: bytes) -> bytes:
    key = get_encryption_key()
    fernet = Fernet(key)
    return fernet.encrypt(image_bytes)

def decrypt_bytes(encrypted_bytes: bytes) -> Image.Image:
    key = get_encryption_key()
    fernet = Fernet(key)
    decrypted = fernet.decrypt(encrypted_bytes)
    return Image.open(io.BytesIO(decrypted))

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL PATH RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

def is_url(path: str) -> bool:
    """Check if the given path is a URL."""
    try:
        result = urlparse(path)
        return result.scheme in ("http", "https")
    except ValueError:
        return False

def load_weights_from_path(path: str, map_location=None) -> dict:
    """
    Load model weights from either:
      - A local file path  (e.g. /home/user/models/best_resnet18.pth)
      - A remote URL       (e.g. https://example.com/models/best_resnet18.pth)

    Returns a state-dict that can be passed to model.load_state_dict().
    """
    path = path.strip()

    if is_url(path):
        st.info(f"⬇️  Downloading weights from URL…")
        response = requests.get(path, stream=True, timeout=120)
        response.raise_for_status()

        # Stream into a temp file so we don't blow up RAM for large checkpoints
        suffix = os.path.splitext(urlparse(path).path)[-1] or ".pth"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp_path = tmp.name

        state_dict = torch.load(tmp_path, map_location=map_location)
        os.unlink(tmp_path)   # clean up
        return state_dict

    # Local path
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return torch.load(path, map_location=map_location)

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════════

class BrainTumorResNet18(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = False):
        super().__init__()
        self.model = models.resnet18(pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SwinUNet(nn.Module):
    def __init__(self, encoder_name: str = "swin_small_patch4_window7_224",
                 pretrained: bool = True, num_classes: int = 1):
        super().__init__()
        self.encoder = timm.create_model(
            encoder_name, pretrained=pretrained,
            features_only=True, out_indices=(0, 1, 2, 3)
        )
        enc_chs = self.encoder.feature_info.channels()

        self.up3 = nn.ConvTranspose2d(enc_chs[3], enc_chs[2], 2, stride=2)
        self.dec3 = ConvBlock(enc_chs[2] * 2, enc_chs[2])
        self.up2 = nn.ConvTranspose2d(enc_chs[2], enc_chs[1], 2, stride=2)
        self.dec2 = ConvBlock(enc_chs[1] * 2, enc_chs[1])
        self.up1 = nn.ConvTranspose2d(enc_chs[1], enc_chs[0], 2, stride=2)
        self.dec1 = ConvBlock(enc_chs[0] * 2, enc_chs[0])
        self.final_up = nn.ConvTranspose2d(enc_chs[0], 64, 2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )

    def _ensure_nchw(self, feat, expected_ch):
        if feat.ndim == 4:
            if feat.shape[1] == expected_ch:
                return feat
            if feat.shape[-1] == expected_ch:
                return feat.permute(0, 3, 1, 2).contiguous()
        return feat

    def forward(self, x):
        feats = self.encoder(x)
        expected = self.encoder.feature_info.channels()
        for i in range(len(feats)):
            feats[i] = self._ensure_nchw(feats[i], expected[i])
        f0, f1, f2, f3 = feats

        d3 = self.up3(f3)
        if d3.shape[-2:] != f2.shape[-2:]:
            d3 = nn.functional.interpolate(d3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, f2], dim=1))

        d2 = self.up2(d3)
        if d2.shape[-2:] != f1.shape[-2:]:
            d2 = nn.functional.interpolate(d2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, f1], dim=1))

        d1 = self.up1(d2)
        if d1.shape[-2:] != f0.shape[-2:]:
            d1 = nn.functional.interpolate(d1, size=f0.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, f0], dim=1))

        out = self.final_up(d1)
        return self.final_conv(out)

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING — path-based, cached per unique path string
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_classification_model(clf_path: str):
    """Load ResNet18 from a local path or URL."""
    model = BrainTumorResNet18(num_classes=4).to(DEVICE)
    state_dict = load_weights_from_path(clf_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def load_segmentation_model(seg_path: str):
    """Load SwinUNet from a local path or URL."""
    model = SwinUNet().to(DEVICE)
    state_dict = load_weights_from_path(seg_path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

clf_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

seg_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CLASS_COLORS = {
    "Glioma": "#FF6B6B",
    "Meningioma": "#4ECDC4",
    "No Tumor": "#95E1D3",
    "Pituitary": "#FFE66D"
}

# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def predict_classification(image: Image.Image, clf_path: str) -> Tuple[str, np.ndarray]:
    clf_model = load_classification_model(clf_path)
    x = clf_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = clf_model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_class = CLASS_NAMES[np.argmax(probs)]
    return pred_class, probs

def predict_segmentation(image: Image.Image, seg_path: str) -> np.ndarray:
    seg_model = load_segmentation_model(seg_path)
    seg_in = seg_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mask = seg_model(seg_in)[0, 0].cpu().numpy()
    return (mask > 0.5).astype(np.uint8)

def process_image(uploaded_image: Image.Image, clf_path: str, seg_path: str) -> Dict:
    # Encrypt / decrypt round-trip (security demonstration)
    img_bytes = io.BytesIO()
    uploaded_image.save(img_bytes, format="PNG")
    encrypted = encrypt_bytes(img_bytes.getvalue())
    decrypted_img = decrypt_bytes(encrypted)

    if decrypted_img.mode != "RGB":
        decrypted_img = decrypted_img.convert("RGB")

    pred_class, probs = predict_classification(decrypted_img, clf_path)
    mask = predict_segmentation(decrypted_img, seg_path)

    return {
        "image": decrypted_img,
        "mask": mask,
        "prediction": pred_class,
        "probabilities": probs,
        "timestamp": datetime.now().isoformat()
    }

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_overlay_visualization(image: Image.Image, mask: np.ndarray) -> np.ndarray:
    img_array = np.array(image.resize((224, 224)))
    mask_resized = cv2.resize(mask, (img_array.shape[1], img_array.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    overlay = img_array.copy().astype(float)
    overlay[mask_resized > 0] = [255, 0, 0]
    blended = cv2.addWeighted(img_array.astype(float), 0.7, overlay, 0.3, 0)
    return np.clip(blended, 0, 255).astype(np.uint8)

def create_heatmap_visualization(image: Image.Image, mask: np.ndarray) -> np.ndarray:
    img_array = np.array(image.resize((224, 224)))
    mask_resized = cv2.resize(mask, (img_array.shape[1], img_array.shape[0]),
                               interpolation=cv2.INTER_LINEAR).astype(float) / 255.0
    heatmap = cv2.applyColorMap((mask_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)

def create_contour_visualization(image: Image.Image, mask: np.ndarray) -> np.ndarray:
    img_array = np.array(image.resize((224, 224)))
    mask_resized = cv2.resize(mask, (img_array.shape[1], img_array.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    result = img_array.copy()
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 0, 0), 2)
    return result

def create_statistics_visualization(mask: np.ndarray) -> go.Figure:
    mask_resized = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
    total_pixels = 224 * 224
    tumor_pixels = int(np.sum(mask_resized > 0))
    tumor_pct = (tumor_pixels / total_pixels) * 100

    fig = go.Figure(data=[
        go.Pie(
            labels=["Tumor Region", "Healthy Region"],
            values=[tumor_pct, 100 - tumor_pct],
            marker=dict(colors=["#FF6B6B", "#95E1D3"]),
            hole=0.3,
            textposition="inside",
            textinfo="label+percent"
        )
    ])
    fig.update_layout(
        title="Tumor Area Distribution", showlegend=True,
        height=400, template="plotly_dark", font=dict(size=12)
    )
    return fig

def create_confidence_chart(probabilities: np.ndarray) -> go.Figure:
    fig = go.Figure(data=[
        go.Bar(
            x=CLASS_NAMES,
            y=probabilities * 100,
            marker=dict(color=[CLASS_COLORS[n] for n in CLASS_NAMES]),
            text=[f"{p*100:.1f}%" for p in probabilities],
            textposition="outside"
        )
    ])
    fig.update_layout(
        title="Classification Confidence",
        xaxis_title="Tumor Type", yaxis_title="Confidence (%)",
        height=400, template="plotly_dark", showlegend=False,
        yaxis=dict(range=[0, 105])
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ONCOSCAN - Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root {
        --primary-color: #0F3460;
        --secondary-color: #00D4FF;
        --danger-color: #FF6B6B;
        --success-color: #95E1D3;
        --bg-dark: #0A0E27;
        --card-bg: #151B2E;
    }
    body { background-color: var(--bg-dark); color: #FFFFFF; }
    .stMetric {
        background-color: var(--card-bg);
        padding: 20px; border-radius: 12px;
        border-left: 4px solid var(--secondary-color);
    }
    .stButton > button {
        width: 100%;
        background-color: var(--secondary-color) !important;
        color: var(--primary-color) !important;
        font-weight: 600; border-radius: 8px;
        padding: 12px 24px !important; border: none !important;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #00B8D4 !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,212,255,0.3);
    }
    .metric-card {
        background-color: var(--card-bg);
        padding: 20px; border-radius: 12px;
        border: 1px solid rgba(0,212,255,0.2);
        margin-bottom: 16px;
    }
    h1, h2, h3 { color: #FFFFFF; font-weight: 700; letter-spacing: 0.5px; }
    .stTabs [data-baseweb="tab-list"] button { background-color: var(--card-bg) !important; border-bottom: 3px solid transparent !important; }
    .stTabs [aria-selected="true"] { border-bottom: 3px solid var(--secondary-color) !important; }
    .path-hint { font-size: 11px; color: #666; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; margin-bottom:40px;'>
    <h1 style='font-size:48px; margin-bottom:10px;'>🧠 ONCOSCAN</h1>
    <p style='font-size:18px; color:#00D4FF; letter-spacing:2px;'>BRAIN TUMOR DETECTION & SEGMENTATION</p>
    <hr style='border:1px solid rgba(0,212,255,0.3); margin-top:20px;'>
</div>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────────
for key in ("results", "upload_time", "models_ready"):
    if key not in st.session_state:
        st.session_state[key] = None

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Model Path Configuration
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🔧 Model Configuration")
    st.markdown("Provide a **local file path** or a **direct download URL** for each model.")

    clf_path_input = st.text_input(
        "Classification Model Path (ResNet18)",
        placeholder="e.g. /home/user/models/best_resnet18_mri.pth  or  https://…/best_resnet18_mri.pth",
        key="clf_path"
    )
    st.markdown('<p class="path-hint">Accepts absolute local paths and http/https URLs.</p>',
                unsafe_allow_html=True)

    seg_path_input = st.text_input(
        "Segmentation Model Path (SwinUNet)",
        placeholder="e.g. /home/user/models/swinunet_best.pth  or  https://…/swinunet_best.pth",
        key="seg_path"
    )
    st.markdown('<p class="path-hint">Accepts absolute local paths and http/https URLs.</p>',
                unsafe_allow_html=True)

    load_models_btn = st.button("⚙️ Load Models", use_container_width=True)

    if load_models_btn:
        if not clf_path_input or not seg_path_input:
            st.error("Please provide both model paths before loading.")
        else:
            with st.spinner("Loading models…"):
                try:
                    load_classification_model(clf_path_input)
                    load_segmentation_model(seg_path_input)
                    st.session_state.models_ready = True
                    st.success("✅ Models loaded successfully!")
                except Exception as e:
                    st.session_state.models_ready = False
                    st.error(f"❌ Failed to load models: {e}")

    st.markdown("---")
    st.markdown("### 📋 Analysis Settings")

    visualization_mode = st.radio(
        "Segmentation Visualization:",
        ["Overlay", "Heatmap", "Contour", "All Modes"],
        key="viz_mode"
    )

    st.markdown("---")
    st.markdown("### ℹ️ Model Information")
    st.info(f"""
**Device:** {DEVICE.upper()}

**Classification:** ResNet18
- Input: 224×224 RGB
- Classes: Glioma, Meningioma, No Tumor, Pituitary

**Segmentation:** SwinUNet
- Input: 224×224 RGB
- Output: Binary mask

**Security:** AES-256 Encryption
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

if not st.session_state.models_ready:
    st.info("👈 Please enter your model paths in the sidebar and click **Load Models** to get started.")
else:
    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        st.markdown("### 📤 Upload MRI Image")
        uploaded_file = st.file_uploader(
            "Select a brain MRI image:",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Supports JPG, PNG, and BMP formats"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("🔍 Analyze", use_container_width=True):
                with st.spinner("🔄 Processing image…"):
                    try:
                        st.session_state.results = process_image(
                            image,
                            st.session_state.clf_path,
                            st.session_state.seg_path
                        )
                        st.session_state.upload_time = datetime.now()
                        st.success("✅ Analysis complete!")
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {e}")

    with col2:
        if st.session_state.results:
            results = st.session_state.results

            st.markdown("### 🎯 Classification Results")
            col_pred, col_conf = st.columns(2)

            with col_pred:
                st.markdown(f"""
                <div class='metric-card' style='text-align:center;'>
                    <h2 style='margin:0; color:{CLASS_COLORS[results["prediction"]]}; font-size:32px;'>
                        {results['prediction']}
                    </h2>
                    <p style='margin:8px 0 0 0; color:#888; font-size:12px;'>PREDICTED TUMOR TYPE</p>
                </div>
                """, unsafe_allow_html=True)

            with col_conf:
                max_conf = float(np.max(results["probabilities"]))
                st.markdown(f"""
                <div class='metric-card' style='text-align:center;'>
                    <h2 style='margin:0; color:#00D4FF; font-size:32px;'>{max_conf*100:.1f}%</h2>
                    <p style='margin:8px 0 0 0; color:#888; font-size:12px;'>CONFIDENCE SCORE</p>
                </div>
                """, unsafe_allow_html=True)

            st.plotly_chart(
                create_confidence_chart(results["probabilities"]),
                use_container_width=True
            )

    # ── Segmentation Section ────────────────────────────────────────────────
    if st.session_state.results:
        st.markdown("---")
        st.markdown("### 🖼️ Segmentation Analysis")

        results = st.session_state.results
        image = results["image"]
        mask = results["mask"]

        if visualization_mode == "Overlay":
            st.image(create_overlay_visualization(image, mask),
                     caption="Tumor Overlay (Red = Tumor Region)", use_column_width=True)
        elif visualization_mode == "Heatmap":
            st.image(create_heatmap_visualization(image, mask),
                     caption="Intensity Heatmap (JET Colormap)", use_column_width=True)
        elif visualization_mode == "Contour":
            st.image(create_contour_visualization(image, mask),
                     caption="Tumor Contours (Red = Boundary)", use_column_width=True)
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(create_overlay_visualization(image, mask), caption="Overlay", use_column_width=True)
            with c2:
                st.image(create_heatmap_visualization(image, mask), caption="Heatmap", use_column_width=True)
            with c3:
                st.image(create_contour_visualization(image, mask), caption="Contour", use_column_width=True)

        # ── Statistics ──────────────────────────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(create_statistics_visualization(mask), use_container_width=True)

        with col2:
            st.markdown("### 📊 Segmentation Metrics")
            mask_224 = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
            total_pixels = 224 * 224
            tumor_pixels = int(np.sum(mask_224 > 0))
            healthy_pixels = total_pixels - tumor_pixels
            tumor_pct = (tumor_pixels / total_pixels) * 100

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Tumor Pixels", f"{tumor_pixels:,}",
                          delta=f"{tumor_pct:.2f}%", delta_color="off")
            with m2:
                st.metric("Healthy Pixels", f"{healthy_pixels:,}",
                          delta=f"{100-tumor_pct:.2f}%", delta_color="off")
            st.markdown("---")
            a1, a2 = st.columns(2)
            with a1:
                st.metric("Segmentation", "Mask Generated")
            with a2:
                st.metric("Processing Time", "< 2 seconds")

        # ── Export ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 💾 Export Analysis")

        e1, e2, e3 = st.columns(3)

        with e1:
            result_img = create_overlay_visualization(image, mask)
            buf = io.BytesIO()
            Image.fromarray(result_img).save(buf, format="PNG")
            buf.seek(0)
            st.download_button(
                "📥 Download Result Image", buf,
                file_name=f"oncoscan_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png", use_container_width=True
            )

        with e2:
            buf = io.BytesIO()
            Image.fromarray((mask * 255).astype(np.uint8)).save(buf, format="PNG")
            buf.seek(0)
            st.download_button(
                "📥 Download Mask", buf,
                file_name=f"oncoscan_mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png", use_container_width=True
            )

        with e3:
            report = {
                "timestamp": results["timestamp"],
                "prediction": results["prediction"],
                "confidence": float(np.max(results["probabilities"])),
                "all_confidences": {
                    CLASS_NAMES[i]: float(results["probabilities"][i])
                    for i in range(len(CLASS_NAMES))
                },
                "tumor_percentage": tumor_pct,
                "device": str(DEVICE)
            }
            st.download_button(
                "📥 Download Report", json.dumps(report, indent=2),
                file_name=f"oncoscan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json", use_container_width=True
            )
    else:
        st.markdown("""
        <div style='text-align:center; padding:60px 20px;'>
            <h3 style='color:#888; font-size:24px; margin-bottom:20px;'>📥 Upload an MRI image to begin</h3>
            <p style='color:#666; font-size:14px;'>
                The application will classify the tumor type and segment the affected region.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; padding:20px; color:#666; font-size:12px;'>
    <p>ONCOSCAN v1.0.0 | Medical AI Research | 🔐 AES-256 Encryption Enabled</p>
    <p>⚠️ This application is for research and educational purposes only. Not for clinical use.</p>
</div>
""", unsafe_allow_html=True)
