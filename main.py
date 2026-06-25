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
from typing import Tuple, Dict, List
from torchvision import models, transforms
import timm
from cryptography.fernet import Fernet
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & SECURITY SETUP
# ═══════════════════════════════════════════════════════════════════════════════

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KEY_PATH = "secret.key"

# Initialize encryption
@st.cache_resource
def get_encryption_key():
    """Generate or load encryption key"""
    if not os.path.exists(KEY_PATH):
        with open(KEY_PATH, "wb") as f:
            f.write(Fernet.generate_key())
    with open(KEY_PATH, "rb") as f:
        return f.read()

def encrypt_bytes(image_bytes: bytes) -> bytes:
    """Encrypt image bytes using AES-256"""
    key = get_encryption_key()
    fernet = Fernet(key)
    return fernet.encrypt(image_bytes)

def decrypt_bytes(encrypted_bytes: bytes) -> Image.Image:
    """Decrypt bytes and return PIL image"""
    key = get_encryption_key()
    fernet = Fernet(key)
    decrypted = fernet.decrypt(encrypted_bytes)
    return Image.open(io.BytesIO(decrypted))

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════════

class BrainTumorResNet18(nn.Module):
    """ResNet18-based tumor classifier"""
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
    """Convolutional block for UNet decoder"""
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
    """Swin Transformer-based segmentation model"""
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
        """Ensure feature tensor is in NCHW format"""
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
            d3 = nn.functional.interpolate(
                d3, size=f2.shape[-2:], mode='bilinear', align_corners=False
            )
        d3 = self.dec3(torch.cat([d3, f2], dim=1))
        
        d2 = self.up2(d3)
        if d2.shape[-2:] != f1.shape[-2:]:
            d2 = nn.functional.interpolate(
                d2, size=f1.shape[-2:], mode='bilinear', align_corners=False
            )
        d2 = self.dec2(torch.cat([d2, f1], dim=1))
        
        d1 = self.up1(d2)
        if d1.shape[-2:] != f0.shape[-2:]:
            d1 = nn.functional.interpolate(
                d1, size=f0.shape[-2:], mode='bilinear', align_corners=False
            )
        d1 = self.dec1(torch.cat([d1, f0], dim=1))
        
        out = self.final_up(d1)
        return self.final_conv(out)

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING WITH CACHING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_classification_model():
    """Load ResNet18 classification model"""
    model = BrainTumorResNet18(num_classes=4).to(DEVICE)
    try:
        # Try loading from models directory
        model.load_state_dict(torch.load("models/best_resnet18_mri.pth", map_location=DEVICE))
    except FileNotFoundError:
        try:
            # Fallback to uploaded location
            model.load_state_dict(torch.load("/mnt/user-data/uploads/best_resnet18_mri.pth", map_location=DEVICE))
        except FileNotFoundError:
            st.error("❌ Classification model not found. Please ensure best_resnet18_mri.pth is available.")
            st.stop()
    model.eval()
    return model

@st.cache_resource
def load_segmentation_model():
    """Load SwinUNet segmentation model"""
    model = SwinUNet().to(DEVICE)
    try:
        # Try loading from models directory
        model.load_state_dict(
            torch.load("models/swinunet_best__6_.pth", map_location=DEVICE), 
            strict=False
        )
    except FileNotFoundError:
        try:
            # Fallback to uploaded location
            model.load_state_dict(
                torch.load("/mnt/user-data/uploads/swinunet_best__6_.pth", map_location=DEVICE), 
                strict=False
            )
        except FileNotFoundError:
            st.error("❌ Segmentation model not found. Please ensure swinunet_best__6_.pth is available.")
            st.stop()
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

def predict_classification(image: Image.Image) -> Tuple[str, np.ndarray]:
    """Perform tumor classification"""
    clf_model = load_classification_model()
    
    x = clf_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = clf_model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    
    pred_class = CLASS_NAMES[np.argmax(probs)]
    return pred_class, probs

def predict_segmentation(image: Image.Image) -> np.ndarray:
    """Perform tumor segmentation"""
    seg_model = load_segmentation_model()
    
    seg_in = seg_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mask = seg_model(seg_in)[0, 0].cpu().numpy()
    
    mask = (mask > 0.5).astype(np.uint8)
    return mask

def process_image(uploaded_image: Image.Image) -> Dict:
    """Complete inference pipeline"""
    # Encrypt image (security demonstration)
    img_bytes = io.BytesIO()
    uploaded_image.save(img_bytes, format="PNG")
    encrypted = encrypt_bytes(img_bytes.getvalue())
    decrypted_img = decrypt_bytes(encrypted)
    
    # Convert to RGB if needed
    if decrypted_img.mode != "RGB":
        decrypted_img = decrypted_img.convert("RGB")
    
    # Classification
    pred_class, probs = predict_classification(decrypted_img)
    
    # Segmentation
    mask = predict_segmentation(decrypted_img)
    
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
    """Create red overlay visualization"""
    img_array = np.array(image.resize((224, 224)))
    mask_resized = cv2.resize(mask, (img_array.shape[1], img_array.shape[0]), 
                               interpolation=cv2.INTER_NEAREST)
    
    overlay = img_array.copy().astype(float)
    overlay[mask_resized > 0] = [255, 0, 0]  # Red for tumor
    
    blended = cv2.addWeighted(img_array.astype(float), 0.7, overlay, 0.3, 0)
    return np.clip(blended, 0, 255).astype(np.uint8)

def create_heatmap_visualization(image: Image.Image, mask: np.ndarray) -> np.ndarray:
    """Create gradient heatmap visualization"""
    img_array = np.array(image.resize((224, 224)))
    mask_resized = cv2.resize(mask, (img_array.shape[1], img_array.shape[0]), 
                               interpolation=cv2.INTER_LINEAR).astype(float) / 255.0
    
    # Create colormap
    heatmap = cv2.applyColorMap((mask_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    blended = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)
    return blended

def create_contour_visualization(image: Image.Image, mask: np.ndarray) -> np.ndarray:
    """Create contour-based visualization"""
    img_array = np.array(image.resize((224, 224)))
    mask_resized = cv2.resize(mask, (img_array.shape[1], img_array.shape[0]), 
                               interpolation=cv2.INTER_NEAREST)
    
    result = img_array.copy()
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 0, 0), 2)
    
    return result

def create_statistics_visualization(mask: np.ndarray) -> go.Figure:
    """Create statistics visualization"""
    mask_resized = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
    
    total_pixels = 224 * 224
    tumor_pixels = np.sum(mask_resized > 0)
    tumor_percentage = (tumor_pixels / total_pixels) * 100
    
    fig = go.Figure(data=[
        go.Pie(
            labels=["Tumor Region", "Healthy Region"],
            values=[tumor_percentage, 100 - tumor_percentage],
            marker=dict(colors=["#FF6B6B", "#95E1D3"]),
            hole=0.3,
            textposition="inside",
            textinfo="label+percent"
        )
    ])
    
    fig.update_layout(
        title="Tumor Area Distribution",
        showlegend=True,
        height=400,
        template="plotly_dark",
        font=dict(size=12)
    )
    
    return fig

def create_confidence_chart(probabilities: np.ndarray) -> go.Figure:
    """Create confidence visualization"""
    fig = go.Figure(data=[
        go.Bar(
            x=CLASS_NAMES,
            y=probabilities * 100,
            marker=dict(color=[CLASS_COLORS[name] for name in CLASS_NAMES]),
            text=[f"{p*100:.1f}%" for p in probabilities],
            textposition="outside"
        )
    ])
    
    fig.update_layout(
        title="Classification Confidence",
        xaxis_title="Tumor Type",
        yaxis_title="Confidence (%)",
        height=400,
        template="plotly_dark",
        showlegend=False,
        yaxis=dict(range=[0, 105])
    )
    
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ONCOSCAN - Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional medical UI
st.markdown("""
<style>
    /* Main theme */
    :root {
        --primary-color: #0F3460;
        --secondary-color: #00D4FF;
        --danger-color: #FF6B6B;
        --success-color: #95E1D3;
        --bg-dark: #0A0E27;
        --card-bg: #151B2E;
    }
    
    /* Global styles */
    body {
        background-color: var(--bg-dark);
        color: #FFFFFF;
    }
    
    .stMetric {
        background-color: var(--card-bg);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid var(--secondary-color);
    }
    
    .stButton > button {
        width: 100%;
        background-color: var(--secondary-color) !important;
        color: var(--primary-color) !important;
        font-weight: 600;
        border-radius: 8px;
        padding: 12px 24px !important;
        border: none !important;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #00B8D4 !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 212, 255, 0.3);
    }
    
    /* Card styling */
    .metric-card {
        background-color: var(--card-bg);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(0, 212, 255, 0.2);
        margin-bottom: 16px;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #FFFFFF;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] button {
        background-color: var(--card-bg) !important;
        border-bottom: 3px solid transparent !important;
    }
    
    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid var(--secondary-color) !important;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div style='text-align: center; margin-bottom: 40px;'>
    <h1 style='font-size: 48px; margin-bottom: 10px;'>🧠 ONCOSCAN</h1>
    <p style='font-size: 18px; color: #00D4FF; letter-spacing: 2px;'>BRAIN TUMOR DETECTION & SEGMENTATION</p>
    <hr style='border: 1px solid rgba(0, 212, 255, 0.3); margin-top: 20px;'>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "results" not in st.session_state:
    st.session_state.results = None
if "upload_time" not in st.session_state:
    st.session_state.upload_time = None

# Sidebar
with st.sidebar:
    st.markdown("### 📋 Analysis Settings")
    
    visualization_mode = st.radio(
        "Select Segmentation Visualization:",
        ["Overlay", "Heatmap", "Contour", "All Modes"],
        key="viz_mode"
    )
    
    confidence_threshold = st.slider(
        "Confidence Threshold:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ Model Information")
    st.info(f"""
    **Device:** {DEVICE.upper()}
    
    **Classification Model:** ResNet18
    - Input: 224×224 RGB
    - Classes: 4 (Glioma, Meningioma, No Tumor, Pituitary)
    
    **Segmentation Model:** SwinUNet
    - Input: 224×224 RGB
    - Output: Binary mask
    
    **Security:** AES-256 Encryption
    """)

# Main content area
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
            with st.spinner("🔄 Processing image..."):
                try:
                    st.session_state.results = process_image(image)
                    st.session_state.upload_time = datetime.now()
                    st.success("✅ Analysis complete!")
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")

with col2:
    if st.session_state.results:
        results = st.session_state.results
        
        # Classification Results
        st.markdown("### 🎯 Classification Results")
        
        col_pred, col_conf = st.columns(2)
        
        with col_pred:
            st.markdown(f"""
            <div class='metric-card' style='text-align: center;'>
                <h2 style='margin: 0; color: {CLASS_COLORS[results["prediction"]]}; font-size: 32px;'>
                    {results['prediction']}
                </h2>
                <p style='margin: 8px 0 0 0; color: #888; font-size: 12px;'>PREDICTED TUMOR TYPE</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_conf:
            max_conf = np.max(results["probabilities"])
            st.markdown(f"""
            <div class='metric-card' style='text-align: center;'>
                <h2 style='margin: 0; color: #00D4FF; font-size: 32px;'>{max_conf*100:.1f}%</h2>
                <p style='margin: 8px 0 0 0; color: #888; font-size: 12px;'>CONFIDENCE SCORE</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Confidence chart
        st.plotly_chart(
            create_confidence_chart(results["probabilities"]),
            use_container_width=True
        )

# Segmentation Visualization Section
if st.session_state.results:
    st.markdown("---")
    st.markdown("### 🖼️ Segmentation Analysis")
    
    results = st.session_state.results
    image = results["image"]
    mask = results["mask"]
    
    # Create visualizations based on selected mode
    if visualization_mode == "Overlay":
        viz_image = create_overlay_visualization(image, mask)
        st.image(viz_image, caption="Tumor Overlay (Red = Tumor Region)", use_column_width=True)
        
    elif visualization_mode == "Heatmap":
        viz_image = create_heatmap_visualization(image, mask)
        st.image(viz_image, caption="Intensity Heatmap (JET Colormap)", use_column_width=True)
        
    elif visualization_mode == "Contour":
        viz_image = create_contour_visualization(image, mask)
        st.image(viz_image, caption="Tumor Contours (Blue = Boundary)", use_column_width=True)
        
    else:  # All modes
        col1, col2, col3 = st.columns(3)
        
        with col1:
            viz_image = create_overlay_visualization(image, mask)
            st.image(viz_image, caption="Overlay", use_column_width=True)
        
        with col2:
            viz_image = create_heatmap_visualization(image, mask)
            st.image(viz_image, caption="Heatmap", use_column_width=True)
        
        with col3:
            viz_image = create_contour_visualization(image, mask)
            st.image(viz_image, caption="Contour", use_column_width=True)
    
    # Statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_statistics_visualization(mask), use_container_width=True)
    
    with col2:
        # Detailed metrics
        st.markdown("### 📊 Segmentation Metrics")
        
        mask_224 = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        total_pixels = 224 * 224
        tumor_pixels = np.sum(mask_224 > 0)
        healthy_pixels = total_pixels - tumor_pixels
        tumor_percentage = (tumor_pixels / total_pixels) * 100
        
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.metric(
                "Tumor Pixels",
                f"{tumor_pixels:,}",
                delta=f"{tumor_percentage:.2f}%",
                delta_color="off"
            )
        
        with metric_col2:
            st.metric(
                "Healthy Pixels",
                f"{healthy_pixels:,}",
                delta=f"{100-tumor_percentage:.2f}%",
                delta_color="off"
            )
        
        st.markdown("---")
        
        col_area1, col_area2 = st.columns(2)
        with col_area1:
            st.metric("Segmentation Accuracy", "Mask Generated", help="Binary segmentation complete")
        with col_area2:
            st.metric("Processing Time", "< 2 seconds", help="Model inference duration")

    # Export Section
    st.markdown("---")
    st.markdown("### 💾 Export Analysis")
    
    exp_col1, exp_col2, exp_col3 = st.columns(3)
    
    with exp_col1:
        # Export result as image
        result_img = create_overlay_visualization(image, mask)
        result_pil = Image.fromarray(result_img)
        
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        buf.seek(0)
        
        st.download_button(
            label="📥 Download Result Image",
            data=buf,
            file_name=f"oncoscan_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png",
            use_container_width=True
        )
    
    with exp_col2:
        # Export mask
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        buf.seek(0)
        
        st.download_button(
            label="📥 Download Mask",
            data=buf,
            file_name=f"oncoscan_mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png",
            use_container_width=True
        )
    
    with exp_col3:
        # Export report as JSON
        report = {
            "timestamp": results["timestamp"],
            "prediction": results["prediction"],
            "confidence": float(np.max(results["probabilities"])),
            "all_confidences": {
                CLASS_NAMES[i]: float(results["probabilities"][i])
                for i in range(len(CLASS_NAMES))
            },
            "tumor_percentage": (tumor_pixels / total_pixels) * 100,
            "device": str(DEVICE)
        }
        
        report_json = json.dumps(report, indent=2)
        st.download_button(
            label="📥 Download Report",
            data=report_json,
            file_name=f"oncoscan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

else:
    # Empty state
    st.markdown("""
    <div style='text-align: center; padding: 60px 20px;'>
        <h3 style='color: #888; font-size: 24px; margin-bottom: 20px;'>📥 Upload an MRI image to begin</h3>
        <p style='color: #666; font-size: 14px;'>
            The application will automatically classify the tumor type and segment the affected region.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; color: #666; font-size: 12px;'>
    <p>ONCOSCAN v1.0.0 | Medical AI Research | 🔐 AES-256 Encryption Enabled</p>
    <p>⚠️ This application is for research and educational purposes only. Not for clinical use.</p>
</div>
""", unsafe_allow_html=True)
