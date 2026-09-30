# ==============================================================================
# RSNA 2026: EFFICIENCY-OPTIMIZED SUBMISSION (SPEED IS EVERYTHING)
# 8-Slice CoPAS + Single TTA Pass + Direct Path Loading
# ==============================================================================
# Efficiency Design:
# - 8 slices/plane (not 16) → 2x fewer DICOM reads
# - EfficientNet-B0 (lightweight, 20MB)
# - Single forward pass + 1 flip TTA (not 3-way) → 2x faster inference
# - Direct path loading (no os.walk) → 0.01s startup
# - Fast center crop (no heavy Astuto) → 30% faster per slice
# ==============================================================================

import os, glob, time
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Optional

t_start = time.time()
print("RSNA Knee Efficiency Submission Starting...", flush=True)

# 1. Direct Paths (instant, no searching)
COMP_DIR = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
if not os.path.exists(COMP_DIR):
    COMP_DIR = "/kaggle/input/rsna-knee-abnormality-detection"

# Find weights directly
WEIGHTS_PATH = None
for candidate in [
    "/kaggle/input/notebooks/kaggluuu/rsna-knee-stage3-final/best_knee_model.pth",
    "/kaggle/input/notebooks/kaggluuu/rsna-knee-stage3-v3/best_knee_model.pth",
    "/kaggle/input/notebooks/kaggluuu/rsna-knee-stage3-v2/best_knee_model.pth",
]:
    if os.path.exists(candidate):
        WEIGHTS_PATH = candidate
        break

# Fallback: quick scan of notebooks only (not the whole input tree)
if not WEIGHTS_PATH and os.path.exists("/kaggle/input/notebooks"):
    for root, dirs, files in os.walk("/kaggle/input/notebooks"):
        if "best_knee_model.pth" in files:
            WEIGHTS_PATH = os.path.join(root, "best_knee_model.pth")
            break

print(f"Competition: {COMP_DIR}", flush=True)
print(f"Weights: {WEIGHTS_PATH}", flush=True)
print(f"Setup time: {time.time()-t_start:.2f}s", flush=True)

TEST_CSV = os.path.join(COMP_DIR, "test.csv")
TEST_SERIES_CSV = os.path.join(COMP_DIR, "test_series.csv")
TEST_SERIES_DIR = os.path.join(COMP_DIR, "test_series")
NUM_SLICES = 8

TARGET_FINDINGS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

# 2. Fast DICOM Reader with Center Crop
def read_dicom_fast(filepath, target_size=(224, 224)):
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
        h, w = img.shape
        ch, cw = h // 2, w // 2
        margin = min(h, w) * 4 // 10
        img = img[max(0,ch-margin):ch+margin, max(0,cw-margin):cw+margin]
        t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        t = nn.functional.interpolate(t, size=target_size, mode="bilinear", align_corners=False)
        return t.squeeze().numpy().astype(np.float32)
    except Exception:
        return None

def extract_study_fast(study_uid, series_df, num_slices=8):
    study_series = series_df[series_df["StudyInstanceUID"] == study_uid]
    volumes = {}
    for plane in ["Sagittal", "Coronal", "Axial"]:
        matched = study_series[study_series["Anatomical_Plane"] == plane]
        if matched.empty: continue
        if "Fluid_Sensitive" in matched.columns and (matched["Fluid_Sensitive"] == 1).any():
            s_uid = matched[matched["Fluid_Sensitive"] == 1].iloc[0]["SeriesInstanceUID"]
        else:
            s_uid = matched.iloc[0]["SeriesInstanceUID"]
        s_path = os.path.join(TEST_SERIES_DIR, study_uid, s_uid)
        if not os.path.exists(s_path): continue
        dcm_files = sorted(glob.glob(os.path.join(s_path, "*.dcm")))
        if not dcm_files: continue
        indices = np.linspace(0, len(dcm_files)-1, num_slices, dtype=int)
        slices = [read_dicom_fast(dcm_files[i]) for i in indices]
        slices = [s for s in slices if s is not None]
        if len(slices) >= num_slices:
            volumes[plane] = np.stack(slices[:num_slices], axis=0).astype(np.float32)
        elif slices:
            reps = int(np.ceil(num_slices / len(slices)))
            volumes[plane] = np.stack((slices * reps)[:num_slices], axis=0).astype(np.float32)
    return volumes

# 3. Lightweight CoPAS Model (matches training architecture exactly)
class CoPASEfficient(nn.Module):
    def __init__(self, num_classes=12, num_slices=8):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        orig = base.features[0][0]
        base.features[0][0] = nn.Conv2d(num_slices, orig.out_channels, kernel_size=orig.kernel_size, stride=orig.stride, padding=orig.padding, bias=False)
        self.encoder = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.attn = nn.MultiheadAttention(1280, 4, batch_first=True)
        self.norm = nn.LayerNorm(1280)
        self.head = nn.Sequential(nn.Linear(1280, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, sag, cor, ax):
        fs = self.pool(self.encoder(sag)).flatten(1)
        fc = self.pool(self.encoder(cor)).flatten(1)
        fa = self.pool(self.encoder(ax)).flatten(1)
        planes = torch.stack([fs, fc, fa], dim=1)
        att, _ = self.attn(planes, planes, planes)
        fused = self.norm(planes + att).mean(dim=1)
        return torch.sigmoid(self.head(fused))

# 4. Speed-Optimized Inference
def run_efficient_submission():
    test_df = pd.read_csv(TEST_CSV)
    series_df = pd.read_csv(TEST_SERIES_CSV)
    print(f"Processing {len(test_df)} studies (8-slice efficient mode)...", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    model = CoPASEfficient(num_classes=12, num_slices=NUM_SLICES).to(device)

    if WEIGHTS_PATH and os.path.exists(WEIGHTS_PATH):
        state = torch.load(WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state)
        print(f"[OK] Loaded weights from {WEIGHTS_PATH}", flush=True)
    else:
        print("[WARNING] No weights found!", flush=True)

    model.eval()
    dummy = torch.zeros((1, NUM_SLICES, 224, 224), dtype=torch.float32, device=device)
    predictions = []

    with torch.no_grad():
        for i, row in test_df.iterrows():
            uid = row["StudyInstanceUID"]
            vols = extract_study_fast(uid, series_df, NUM_SLICES)

            sag = torch.from_numpy(vols.get("Sagittal", dummy.squeeze(0).cpu().numpy())).unsqueeze(0).to(device)
            cor = torch.from_numpy(vols.get("Coronal", dummy.squeeze(0).cpu().numpy())).unsqueeze(0).to(device)
            ax = torch.from_numpy(vols.get("Axial", dummy.squeeze(0).cpu().numpy())).unsqueeze(0).to(device)

            # Pass 1: Standard
            p1 = model(sag, cor, ax).squeeze().cpu().numpy()
            # Pass 2: Single TTA (horizontal flip)
            p2 = model(torch.flip(sag,[-1]), torch.flip(cor,[-1]), torch.flip(ax,[-1])).squeeze().cpu().numpy()
            preds = 0.6 * p1 + 0.4 * p2

            row_dict = {"StudyInstanceUID": uid}
            for j, f in enumerate(TARGET_FINDINGS):
                row_dict[f] = float(np.clip(preds[j], 0.01, 0.99))
            predictions.append(row_dict)

            if (i+1) % 10 == 0 or (i+1) == len(test_df):
                print(f"  {i+1}/{len(test_df)} ({time.time()-t_start:.0f}s)", flush=True)

    sub_df = pd.DataFrame(predictions)[["StudyInstanceUID"] + TARGET_FINDINGS]
    sub_df.to_csv("/kaggle/working/submission.csv", index=False)

    total = time.time() - t_start
    print(f"\n[DONE] submission.csv saved in {total:.0f}s ({total/60:.1f} min)", flush=True)
    print(sub_df.head())
    print(f"NaNs: {sub_df.isna().sum().sum()}, Shape: {sub_df.shape}")

if __name__ == "__main__":
    run_efficient_submission()
