# ==============================================================================
# RSNA 2026 KNEE ABNORMALITY DETECTION: FINAL SUBMISSION V3
# TRAINED CoPAS V3 + ANATOMICAL ROI CROPPER + TEST-TIME AUGMENTATION (TTA)
# ==============================================================================
# Target Objective: Maximize Macro-ROC-AUC toward 0.961
# Enhancements:
# 1. Loads real-label supervised CoPAS weights (from rsna-knee-stage3-v3).
# 2. Astuto et al. Knee Joint ROI Cropper to match training distribution.
# 3. Test-Time Augmentation (TTA): Averages original and flipped volume passes.
# 4. Strict probability bounds [0.01, 0.99] with zero NaNs.
# ==============================================================================

import os
import glob
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, List, Optional

print("Locating competition data and trained V3 model weights...", flush=True)

# 1. Path Discovery
COMP_DIR = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
if not os.path.exists(COMP_DIR):
    COMP_DIR = "/kaggle/input/rsna-knee-abnormality-detection"

WEIGHTS_PATH = None
if os.path.exists("/kaggle/input"):
    for root, dirs, files in os.walk("/kaggle/input"):
        if "best_knee_model.pth" in files and "v3" in root.lower():
            WEIGHTS_PATH = os.path.join(root, "best_knee_model.pth")
            break
    if not WEIGHTS_PATH:
        # General search if folder name doesn't contain 'v3'
        for root, dirs, files in os.walk("/kaggle/input"):
            if "best_knee_model.pth" in files:
                WEIGHTS_PATH = os.path.join(root, "best_knee_model.pth")
                break

print(f"Competition directory: {COMP_DIR}", flush=True)
print(f"Trained V3 weights path: {WEIGHTS_PATH}", flush=True)

TEST_CSV = os.path.join(COMP_DIR, "test.csv")
TEST_SERIES_CSV = os.path.join(COMP_DIR, "test_series.csv")
SAMPLE_SUB_CSV = os.path.join(COMP_DIR, "sample_submission.csv")
TEST_SERIES_DIR = os.path.join(COMP_DIR, "test_series")
OUTPUT_SUB_CSV = "/kaggle/working/submission.csv"

TARGET_FINDINGS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

# 2. Knee Joint ROI Cropper (Astuto et al. adaptation)
def crop_knee_joint(img: np.ndarray, margin_ratio: float = 0.08) -> np.ndarray:
    threshold = np.percentile(img, 15)
    binary_mask = img > max(threshold, 0.05)

    row_sums = np.sum(binary_mask, axis=1)
    col_sums = np.sum(binary_mask, axis=0)

    rows = np.where(row_sums > 0.05 * img.shape[1])[0]
    cols = np.where(col_sums > 0.05 * img.shape[0])[0]

    h, w = img.shape
    if len(rows) < 10 or len(cols) < 10:
        return img[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]

    h_margin = int(h * margin_ratio)
    w_margin = int(w * margin_ratio)
    r_min = max(0, rows[0] - h_margin)
    r_max = min(h, rows[-1] + h_margin)
    c_min = max(0, cols[0] - w_margin)
    c_max = min(w, cols[-1] + w_margin)

    return img[r_min:r_max, c_min:c_max]

def read_dicom_slice(filepath: str, target_size=(224, 224)) -> Optional[np.ndarray]:
    try:
        dcm = pydicom.dcmread(filepath, force=True)
        img = dcm.pixel_array.astype(np.float32)
        slope = float(getattr(dcm, "RescaleSlope", 1.0))
        intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
        img = img * slope + intercept
        
        p1, p99 = np.percentile(img, (1, 99))
        if p99 > p1:
            img = np.clip(img, p1, p99)
            img = (img - p1) / (p99 - p1)
        else:
            img = np.zeros_like(img)

        # Apply same ROI crop as training
        cropped = crop_knee_joint(img)

        t_img = torch.from_numpy(cropped).float().unsqueeze(0).unsqueeze(0)
        t_resized = nn.functional.interpolate(t_img, size=target_size, mode="bilinear", align_corners=False)
        return t_resized.squeeze().cpu().numpy().astype(np.float32)
    except Exception:
        return None

def extract_study_slices(study_uid: str, series_df: pd.DataFrame, num_slices: int = 16) -> Dict[str, np.ndarray]:
    study_series = series_df[series_df["StudyInstanceUID"] == study_uid]
    plane_volumes = {}

    for plane in ["Sagittal", "Coronal", "Axial"]:
        matched = study_series[study_series["Anatomical_Plane"] == plane]
        if matched.empty:
            continue
        
        if "Fluid_Sensitive" in matched.columns and (matched["Fluid_Sensitive"] == 1).any():
            chosen_series_uid = matched[matched["Fluid_Sensitive"] == 1].iloc[0]["SeriesInstanceUID"]
        else:
            chosen_series_uid = matched.iloc[0]["SeriesInstanceUID"]

        series_path = os.path.join(TEST_SERIES_DIR, study_uid, chosen_series_uid)
        if not os.path.exists(series_path):
            continue

        dcm_files = sorted(glob.glob(os.path.join(series_path, "*.dcm")))
        total_slices = len(dcm_files)
        if total_slices == 0:
            continue

        if total_slices >= num_slices:
            margin = (total_slices - num_slices) // 2
            indices = np.linspace(margin, total_slices - margin - 1, num_slices, dtype=int)
        else:
            indices = np.linspace(0, total_slices - 1, num_slices, dtype=int)

        slices = []
        for idx in indices:
            sl = read_dicom_slice(dcm_files[idx])
            if sl is not None:
                slices.append(sl)

        if len(slices) == num_slices:
            plane_volumes[plane] = np.stack(slices, axis=0).astype(np.float32)

    return plane_volumes

class CoPASMultiPlaneModel(nn.Module):
    def __init__(self, num_classes=12, num_slices=16):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        orig_conv = base.features[0][0]
        base.features[0][0] = nn.Conv2d(
            num_slices, orig_conv.out_channels, 
            kernel_size=orig_conv.kernel_size, 
            stride=orig_conv.stride, 
            padding=orig_conv.padding, 
            bias=False
        )
        self.encoder = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = 1280

        self.plane_attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(feat_dim)

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, sag, cor, ax):
        f_sag = self.pool(self.encoder(sag)).flatten(1)
        f_cor = self.pool(self.encoder(cor)).flatten(1)
        f_ax = self.pool(self.encoder(ax)).flatten(1)

        planes = torch.stack([f_sag, f_cor, f_ax], dim=1)
        attn_out, _ = self.plane_attn(planes, planes, planes)
        fused = self.norm(planes + attn_out).mean(dim=1)

        logits = self.classifier(fused)
        return torch.sigmoid(logits)

# 4. Main Inference with Test-Time Augmentation (TTA)
def run_submission_v3():
    test_df = pd.read_csv(TEST_CSV)
    series_df = pd.read_csv(TEST_SERIES_CSV)
    print(f"Processing {len(test_df)} test studies with TTA...", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}", flush=True)

    model = CoPASMultiPlaneModel(num_classes=len(TARGET_FINDINGS)).to(device)
    
    if WEIGHTS_PATH and os.path.exists(WEIGHTS_PATH):
        print(f"Loading trained V3 weights from: {WEIGHTS_PATH}", flush=True)
        state_dict = torch.load(WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print("[SUCCESS] Loaded trained CoPAS V3 model weights!", flush=True)
    else:
        print("[WARNING] Trained weights not found. Using untrained fallback.", flush=True)

    model.eval()

    predictions = []
    dummy_vol = torch.zeros((1, 16, 224, 224), dtype=torch.float32, device=device)

    with torch.no_grad():
        for i, row in test_df.iterrows():
            study_uid = row["StudyInstanceUID"]
            vols = extract_study_slices(study_uid, series_df, num_slices=16)

            sag = torch.from_numpy(vols["Sagittal"]).unsqueeze(0).to(device) if "Sagittal" in vols else dummy_vol
            cor = torch.from_numpy(vols["Coronal"]).unsqueeze(0).to(device) if "Coronal" in vols else dummy_vol
            ax = torch.from_numpy(vols["Axial"]).unsqueeze(0).to(device) if "Axial" in vols else dummy_vol

            # Pass 1: Standard Orientation
            preds_orig = model(sag, cor, ax).squeeze().cpu().numpy()

            # Pass 2: Test-Time Augmentation (Horizontal Flip)
            sag_flip = torch.flip(sag, dims=[-1])
            cor_flip = torch.flip(cor, dims=[-1])
            ax_flip = torch.flip(ax, dims=[-1])
            preds_flip = model(sag_flip, cor_flip, ax_flip).squeeze().cpu().numpy()

            # Ensemble prediction across original and augmented passes
            preds = 0.5 * preds_orig + 0.5 * preds_flip

            row_dict = {"StudyInstanceUID": study_uid}
            for idx, target in enumerate(TARGET_FINDINGS):
                row_dict[target] = float(np.clip(preds[idx], 0.01, 0.99))

            predictions.append(row_dict)

            if (i + 1) % 10 == 0 or (i + 1) == len(test_df):
                print(f"Processed {i + 1}/{len(test_df)} studies", flush=True)

    sub_df = pd.DataFrame(predictions)
    cols = ["StudyInstanceUID"] + TARGET_FINDINGS
    sub_df = sub_df[cols]

    sub_df.to_csv(OUTPUT_SUB_CSV, index=False)
    print(f"\n[FINAL SUCCESS] Saved submission to: {OUTPUT_SUB_CSV}", flush=True)
    print(sub_df.head(), flush=True)
    print("Checking NaNs:", sub_df.isna().sum().sum(), flush=True)
    print("Shape:", sub_df.shape, flush=True)

if __name__ == "__main__":
    run_submission_v3()
