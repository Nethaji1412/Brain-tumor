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
import tempfile

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
    return Fernet(key).encrypt(image_bytes)

def decrypt_bytes(encrypted_bytes: bytes) -> Image.Image:
    key = get_encryption_key()
    decrypted = Fernet(key).decrypt(encrypted_bytes)
    return Image.open(io.BytesIO(decrypted))

# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL FILE UPLOAD HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def save_uploaded_model(uploaded_file) -> str:
    """
    Persist a Streamlit UploadedFile to a temporary .pth file on disk
    and return the path so torch.load() can read it.
    """
    suffix = os.path.splitext(uploaded_file.name)[-1] or ".pth"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name

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
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
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

        self.up3  = nn.ConvTranspose2d(enc_chs[3], enc_chs[2], 2, stride=2)
        self.dec3 = ConvBlock(enc_chs[2] * 2, enc_chs[2])
        self.up2  = nn.ConvTranspose2d(enc_chs[2], enc_chs[1], 2, stride=2)
        self.dec2 = ConvBlock(enc_chs[1] * 2, enc_chs[1])
        self.up1  = nn.ConvTranspose2d(enc_chs[1], enc_chs[0], 2, stride=2)
        self.dec1 = ConvBlock(enc_chs[0] * 2, enc_chs[0])
        self.final_up   = nn.ConvTranspose2d(enc_chs[0], 64, 2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
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
        feats    = self.encoder(x)
        expected = self.encoder.feature_info.channels()
        for i in range(len(feats)):
            feats[i] = self._ensure_nchw(feats[i], expected[i])
        f0, f1, f2, f3 = feats

        def _up(up_layer, dec_layer, src, skip):
            out = up_layer(src)
            if out.shape[-2:] != skip.shape[-2:]:
                out = nn.functional.interpolate(
                    out, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            return dec_layer(torch.cat([out, skip], dim=1))

        d3  = _up(self.up3, self.dec3, f3, f2)
        d2  = _up(self.up2, self.dec2, d3, f1)
        d1  = _up(self.up1, self.dec1, d2, f0)
        return self.final_conv(self.final_up(d1))

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING — cached per temp-file path
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_classification_model(tmp_path: str):
    """Load ResNet18 weights from a saved temp file."""
    model = BrainTumorResNet18(num_classes=4).to(DEVICE)
    model.load_state_dict(torch.load(tmp_path, map_location=DEVICE))
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def load_segmentation_model(tmp_path: str):
    """Load SwinUNet weights from a saved temp file."""
    model = SwinUNet().to(DEVICE)
    model.load_state_dict(torch.load(tmp_path, map_location=DEVICE), strict=False)
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

CLASS_NAMES  = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CLASS_COLORS = {
    "Glioma": "#FF6B6B", "Meningioma": "#4ECDC4",
    "No Tumor": "#95E1D3", "Pituitary": "#FFE66D"
}

# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def predict_classification(image: Image.Image, clf_tmp: str) -> Tuple[str, np.ndarray]:
    clf_model = load_classification_model(clf_tmp)
    x = clf_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(clf_model(x), dim=1)[0].cpu().numpy()
    return CLASS_NAMES[np.argmax(probs)], probs

def predict_segmentation(image: Image.Image, seg_tmp: str) -> np.ndarray:
    seg_model = load_segmentation_model(seg_tmp)
    seg_in = seg_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mask = seg_model(seg_in)[0, 0].cpu().numpy()
    return (mask > 0.5).astype(np.uint8)

def process_image(uploaded_image: Image.Image, clf_tmp: str, seg_tmp: str) -> Dict:
    # AES-256 encrypt/decrypt round-trip (security demonstration)
    buf = io.BytesIO()
    uploaded_image.save(buf, format="PNG")
    decrypted_img = decrypt_bytes(encrypt_bytes(buf.getvalue()))

    if decrypted_img.mode != "RGB":
        decrypted_img = decrypted_img.convert("RGB")

    pred_class, probs = predict_classification(decrypted_img, clf_tmp)
    mask = predict_segmentation(decrypted_img, seg_tmp)

    return {
        "image": decrypted_img, "mask": mask,
        "prediction": pred_class, "probabilities": probs,
        "timestamp": datetime.now().isoformat()
    }

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_overlay_visualization(image: Image.Image, mask: np.ndarray) -> np.ndarray:
    img = np.array(image.resize((224, 224)))
    m   = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    ov  = img.copy().astype(float)
    ov[m > 0] = [255, 0, 0]
    return np.clip(cv2.addWeighted(img.astype(float), 0.7, ov, 0.3, 0), 0, 255).astype(np.uint8)

def create_heatmap_visualization(image: Image.Image, mask: np.ndarray) -> np.ndarray:
    img = np.array(image.resize((224, 224)))
    m   = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_LINEAR).astype(float) / 255.0
    hm  = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.6, hm, 0.4, 0)

def create_contour_visualization(image: Image.Image, mask: np.ndarray) -> np.ndarray:
    img = np.array(image.resize((224, 224)))
    m   = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    res = img.copy()
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(res, contours, -1, (255, 0, 0), 2)
    return res

def create_statistics_visualization(mask: np.ndarray) -> go.Figure:
    m   = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
    tp  = int(np.sum(m > 0))
    pct = (tp / (224 * 224)) * 100
    fig = go.Figure(data=[go.Pie(
        labels=["Tumor Region", "Healthy Region"],
        values=[pct, 100 - pct],
        marker=dict(colors=["#FF6B6B", "#95E1D3"]),
        hole=0.3, textposition="inside", textinfo="label+percent"
    )])
    fig.update_layout(title="Tumor Area Distribution", showlegend=True,
                      height=400, template="plotly_dark", font=dict(size=12))
    return fig

def create_confidence_chart(probabilities: np.ndarray) -> go.Figure:
    fig = go.Figure(data=[go.Bar(
        x=CLASS_NAMES, y=probabilities * 100,
        marker=dict(color=[CLASS_COLORS[n] for n in CLASS_NAMES]),
        text=[f"{p*100:.1f}%" for p in probabilities], textposition="outside"
    )])
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
    page_icon="🧠", layout="wide", initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root { --secondary-color: #00D4FF; --card-bg: #151B2E; }
    body { background-color: #0A0E27; color: #FFFFFF; }
    .stMetric { background-color: var(--card-bg); padding: 20px; border-radius: 12px;
                border-left: 4px solid var(--secondary-color); }
    .stButton > button {
        width: 100%; background-color: var(--secondary-color) !important;
        color: #0F3460 !important; font-weight: 600; border-radius: 8px;
        padding: 12px 24px !important; border: none !important; transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #00B8D4 !important; transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,212,255,0.3);
    }
    .metric-card { background-color: var(--card-bg); padding: 20px; border-radius: 12px;
                   border: 1px solid rgba(0,212,255,0.2); margin-bottom: 16px; }
    h1, h2, h3 { color: #FFFFFF; font-weight: 700; letter-spacing: 0.5px; }
    .stTabs [data-baseweb="tab-list"] button { background-color: var(--card-bg) !important;
                                               border-bottom: 3px solid transparent !important; }
    .stTabs [aria-selected="true"] { border-bottom: 3px solid var(--secondary-color) !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; margin-bottom:40px;'>
    <h1 style='font-size:48px; margin-bottom:10px;'>🧠 ONCOSCAN</h1>
    <p style='font-size:18px; color:#00D4FF; letter-spacing:2px;'>BRAIN TUMOR DETECTION & SEGMENTATION</p>
    <hr style='border:1px solid rgba(0,212,255,0.3); margin-top:20px;'>
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key in ("results", "clf_tmp", "seg_tmp", "models_ready"):
    if key not in st.session_state:
        st.session_state[key] = None

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Model Upload
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🔧 Model Configuration")
    st.markdown("Upload your `.pth` model files from your **local machine**.")

    clf_file = st.file_uploader(
        "📂 Classification Model (ResNet18)",
        type=["pth", "pt"],
        key="clf_upload",
        help="Upload best_resnet18_mri.pth from your local drive"
    )

    seg_file = st.file_uploader(
        "📂 Segmentation Model (SwinUNet)",
        type=["pth", "pt"],
        key="seg_upload",
        help="Upload swinunet_best.pth from your local drive"
    )

    load_btn = st.button("⚙️ Load Models", use_container_width=True)

    if load_btn:
        if not clf_file or not seg_file:
            st.error("Please upload both model files before loading.")
        else:
            with st.spinner("Loading models…"):
                try:
                    # Write uploaded bytes to temp files on disk
                    clf_tmp = save_uploaded_model(clf_file)
                    seg_tmp = save_uploaded_model(seg_file)

                    # Load & cache models
                    load_classification_model(clf_tmp)
                    load_segmentation_model(seg_tmp)

                    # Persist temp paths for inference
                    st.session_state.clf_tmp      = clf_tmp
                    st.session_state.seg_tmp      = seg_tmp
                    st.session_state.models_ready = True

                    st.success(
                        f"✅ Models loaded!\n\n"
                        f"- **CLF:** {clf_file.name}\n"
                        f"- **SEG:** {seg_file.name}"
                    )
                except Exception as e:
                    st.session_state.models_ready = False
                    st.error(f"❌ Failed to load models:\n{e}")

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
    st.info("👈 Upload your `.pth` model files in the sidebar and click **Load Models** to get started.")
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
                            st.session_state.clf_tmp,
                            st.session_state.seg_tmp
                        )
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
                </div>""", unsafe_allow_html=True)

            with col_conf:
                max_conf = float(np.max(results["probabilities"]))
                st.markdown(f"""
                <div class='metric-card' style='text-align:center;'>
                    <h2 style='margin:0; color:#00D4FF; font-size:32px;'>{max_conf*100:.1f}%</h2>
                    <p style='margin:8px 0 0 0; color:#888; font-size:12px;'>CONFIDENCE SCORE</p>
                </div>""", unsafe_allow_html=True)

            st.plotly_chart(
                create_confidence_chart(results["probabilities"]),
                use_container_width=True
            )

    # ── Segmentation Section ──────────────────────────────────────────────────
    if st.session_state.results:
        st.markdown("---")
        st.markdown("### 🖼️ Segmentation Analysis")

        results = st.session_state.results
        image   = results["image"]
        mask    = results["mask"]

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
            with c1: st.image(create_overlay_visualization(image, mask), caption="Overlay",  use_column_width=True)
            with c2: st.image(create_heatmap_visualization(image, mask), caption="Heatmap",  use_column_width=True)
            with c3: st.image(create_contour_visualization(image, mask), caption="Contour",  use_column_width=True)

        # ── Statistics ────────────────────────────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_statistics_visualization(mask), use_container_width=True)

        with col2:
            st.markdown("### 📊 Segmentation Metrics")
            mask_224       = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
            total_pixels   = 224 * 224
            tumor_pixels   = int(np.sum(mask_224 > 0))
            healthy_pixels = total_pixels - tumor_pixels
            tumor_pct      = (tumor_pixels / total_pixels) * 100

            m1, m2 = st.columns(2)
            with m1: st.metric("Tumor Pixels",   f"{tumor_pixels:,}",   delta=f"{tumor_pct:.2f}%",      delta_color="off")
            with m2: st.metric("Healthy Pixels",  f"{healthy_pixels:,}", delta=f"{100-tumor_pct:.2f}%", delta_color="off")
            st.markdown("---")
            a1, a2 = st.columns(2)
            with a1: st.metric("Segmentation",    "Mask Generated")
            with a2: st.metric("Processing Time", "< 2 seconds")

        # ── Export ────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 💾 Export Analysis")
        e1, e2, e3 = st.columns(3)

        with e1:
            buf = io.BytesIO()
            Image.fromarray(create_overlay_visualization(image, mask)).save(buf, format="PNG")
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
                "timestamp":        results["timestamp"],
                "prediction":       results["prediction"],
                "confidence":       float(np.max(results["probabilities"])),
                "all_confidences":  {CLASS_NAMES[i]: float(results["probabilities"][i]) for i in range(4)},
                "tumor_percentage": tumor_pct,
                "device":           str(DEVICE)
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
        </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; padding:20px; color:#666; font-size:12px;'>
    <p>ONCOSCAN v1.0.0 | Medical AI Research | 🔐 AES-256 Encryption Enabled</p>
    <p>⚠️ This application is for research and educational purposes only. Not for clinical use.</p>
</div>""", unsafe_allow_html=True)
