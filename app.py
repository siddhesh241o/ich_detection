import streamlit as st
import pydicom
import numpy as np
import cv2
import os
from pathlib import Path
import torch
import torch.nn as nn
from ultralytics import YOLO
import __main__

# =================================================================
# 1. CUSTOM MODEL COMPONENTS
# =================================================================

class SimAM(nn.Module):
    """Simple Attention Mechanism layer used in VoluYOLO Ultimate."""
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_sq = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_sq / (4 * (x_minus_mu_sq.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)

# Critical: Inject SimAM into the main namespace for the YOLO loader
setattr(__main__, "SimAM", SimAM)

# =================================================================
# 2. DATA ENGINEERING ENGINE
# =================================================================

class DataEngine:
    @staticmethod
    def apply_window(ds, wl=40, ww=80):
        """
        Clinical-grade windowing matching your training preprocessing.
        ds: The loaded pydicom dataset object
        """
        # 1. Get raw pixels
        img = ds.pixel_array.astype(float)

        # 2. Convert to Hounsfield Units (HU)
        # This is the step that was missing and likely caused the 'off' look
        slope = getattr(ds, 'RescaleSlope', 1)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        img = (img * slope) + intercept

        # 3. Apply Windowing (Brain Window: WL=40, WW=80)
        img_min = wl - ww // 2
        img_max = wl + ww // 2
        img = np.clip(img, img_min, img_max)

        # 4. Normalize to 0-255
        img = ((img - img_min) / (img_max - img_min)) * 255.0
        return img.astype(np.uint8)

    def create_stack(self, slices, idx):
        """Creates 2.5D Stack with clinical HU scaling"""
        prev_idx = max(0, idx - 1)
        next_idx = min(len(slices) - 1, idx + 1)
        
        # Pass the full dataset object to the new window function
        ch1 = self.apply_window(slices[prev_idx])
        ch2 = self.apply_window(slices[idx])
        ch3 = self.apply_window(slices[next_idx])
        
        stack = np.dstack((ch1, ch2, ch3))
        return cv2.resize(stack, (640, 640))
# =================================================================
# 3. STREAMLIT TRIAGE APPLICATION
# =================================================================

@st.cache_resource
def load_voluyolo(model_path):
    """Loads the custom YOLOv11 model with SimAM and P2 Head."""
    return YOLO(model_path)

def main():
    st.set_page_config(page_title="VoluYOLO Triage", layout="wide", page_icon="🧠")
    
    st.title("🧠 VoluYOLO Ultimate: Automated Triage")
    st.info("System optimized for detecting tiny hemorrhages using P2 Micro-Head & SimAM Attention.")
    st.markdown("---")

    # Load resources
    try:
        model_path = os.getenv("VOLUYOLO_MODEL_PATH", "./best.pt")
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. "
                "Set VOLUYOLO_MODEL_PATH or place best.pt in the project root."
            )
        model = load_voluyolo(model_path)
        data_engine = DataEngine()
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return

    # Sidebar: Setup and Upload
    st.sidebar.header("📂 Patient Data")
    uploaded_files = st.sidebar.file_uploader("Upload DICOM Volume", accept_multiple_files=True)
    conf_thresh = st.sidebar.slider("AI Confidence Threshold", 0.1, 1.0, 0.25)
    
    if uploaded_files:
        # 1. Read and Sort Slices by Z-axis position
        slices = [pydicom.dcmread(f) for f in uploaded_files]
        slices.sort(key=lambda x: float(getattr(x, 'ImagePositionPatient', [0,0,0])[2]))
        
        st.sidebar.success(f"Loaded {len(slices)} slices.")

        best_slice_data = None
        max_conf = 0.0

        # 2. Automated Volume Scanning (Triage Mode)
        with st.spinner('Analyzing volume for the most critical finding...'):
            for i in range(len(slices)):
                stack = data_engine.create_stack(slices, i)
                results = model.predict(source=stack, conf=conf_thresh, imgsz=640, verbose=False)[0]
                
                if len(results.boxes) > 0:
                    # Identify the peak confidence score for this slice
                    current_max = results.boxes.conf.max().item()
                    
                    # Store only if this is the "most positive" slice found so far
                    if current_max > max_conf:
                        max_conf = current_max
                        best_slice_data = {
                            "index": i,
                            "img": results.plot(),
                            "conf": current_max,
                            "count": len(results.boxes),
                            "inference_time": results.speed['inference']
                        }

        # 3. UI Display: Only the Top Result
        if best_slice_data:
            st.warning(f"🚨 CRITICAL FINDING IDENTIFIED: Slice {best_slice_data['index']}")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(best_slice_data['img'], caption="Highest Confidence Detection (2.5D Stack)", use_container_width=True)
            
            with col2:
                st.metric("Detection Confidence", f"{best_slice_data['conf']:.2%}")
                st.metric("Total Regions Flagged", best_slice_data['count'])
                st.metric("Inference Latency", f"{best_slice_data['inference_time']:.2f} ms")
                
                st.markdown("---")
                st.write("**Clinical Note:**")
                st.caption("This slice was automatically selected as the most significant finding. Please verify with the original 2D window.")
                
                # Show the original non-annotated slice for doctor's comparison
                orig_2d = data_engine.apply_window(slices[best_slice_data['index']])
                st.image(orig_2d, caption="Original Brain Window", use_container_width=True)
        else:
            st.success("✅ Analysis Complete: No intracranial hemorrhage detected in this volume.")

if __name__ == "__main__":
    main()