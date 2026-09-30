# ==============================================================================
# RSNA 2026 KNEE ABNORMALITY DETECTION: STAGE 4 V5 SUBMISSION
# DUAL-CONTRAST EXTRACTOR + CoPAS EFFICIENTNET-B2 + TEST-TIME AUGMENTATION (TTA)
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

print("Locating competition data and trained V5 model weights...", flush=True)

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
        for root, dirs, files in os.walk("/kaggle/input"):
            if "best_knee_model.pth" in files:
                WEIGHTS_PATH = os.path.join(root, "best_knee_model.pth")
                break

print(f"Competition directory: {COMP_DIR}", flush=True)
print(f"Trained V5 weights path: {WEIGHTS_PATH}", flush=True)

TEST_CSV = os.path.join(COMP_DIR, "test.csv")
TEST_SERIES_CSV = os.path.join(COMP_DIR, "test_series.csv")
TEST_SERIES_DIR = os.path.join(COMP_DIR, "test_series")
OUTPUT_SUB_CSV = "/kaggle/working/submission.csv"

TARGET_FINDINGS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

# 2. Knee Joint ROI Cropper (Astuto et al.)
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

        cropped = crop_knee_joint(img)
        t_img = torch.from_numpy(cropped).float().unsqueeze(0).unsqueeze(0)
        t_resized = nn.functional.interpolate(t_img, size=target_size, mode="bilinear", align_corners=False)
        return t_resized.squeeze().cpu().numpy().astype(np.float32)
    except Exception:
        return None

# Dual-contrast slice extraction
def extract_dual_contrast_slices(study_uid: str, series_df: pd.DataFrame, num_slices: int = 16) -> Dict[str, np.ndarray]:
    study_series = series_df[series_df["StudyInstanceUID"] == study_uid]
    plane_volumes = {}

    for plane in ["Sagittal", "Coronal", "Axial"]:
        matched = study_series[study_series["Anatomical_Plane"] == plane]
        if matched.empty:
            continue

        fluid_matches = matched[matched.get("Fluid_Sensitive", 0) == 1]
        non_fluid_matches = matched[matched.get("Fluid_Sensitive", 0) == 0]
        has_both = not fluid_matches.empty and not non_fluid_matches.empty

        slices_list = []
        if has_both:
            for sub_df, count in [(fluid_matches, num_slices // 2), (non_fluid_matches, num_slices // 2)]:
                s_uid = sub_df.iloc[0]["SeriesInstanceUID"]
                s_path = os.path.join(TEST_SERIES_DIR, study_uid, s_uid)
                dcm_files = sorted(glob.glob(os.path.join(s_path, "*.dcm")))
                if len(dcm_files) > 0:
                    indices = np.linspace(0, len(dcm_files) - 1, count, dtype=int)
                    for idx in indices:
                        sl = read_dicom_slice(dcm_files[idx])
                        if sl is not None:
                            slices_list.append(sl)
        else:
            s_uid = matched.iloc[0]["SeriesInstanceUID"]
            s_path = os.path.join(TEST_SERIES_DIR, study_uid, s_uid)
            dcm_files = sorted(glob.glob(os.path.join(s_path, "*.dcm")))
            if len(dcm_files) > 0:
                indices = np.linspace(0, len(dcm_files) - 1, num_slices, dtype=int)
                for idx in indices:
                    sl = read_dicom_slice(dcm_files[idx])
                    if sl is not None:
                        slices_list.append(sl)

        if len(slices_list) >= num_slices:
            plane_volumes[plane] = np.stack(slices_list[:num_slices], axis=0).astype(np.float32)
        elif len(slices_list) > 0:
            reps = int(np.ceil(num_slices / len(slices_list)))
            tiled = (slices_list * reps)[:num_slices]
            plane_volumes[plane] = np.stack(tiled, axis=0).astype(np.float32)

    return plane_volumes

# 3. Model Architecture (Offline compatible, EfficientNet-B2)
class CoPASMultiPlaneB2Model(nn.Module):
    def __init__(self, num_classes=12, num_slices=16):
        super().__init__()
        base = models.efficientnet_b2(weights=None)
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
        feat_dim = 1408

        self.plane_attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(feat_dim)

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 384),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(384, num_classes)
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

# 4. Inference Engine with TTA
def run_submission_v5():
    test_df = pd.read_csv(TEST_CSV)
    series_df = pd.read_csv(TEST_SERIES_CSV)
    print(f"Processing {len(test_df)} test studies with TTA...", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}", flush=True)

    model = CoPASMultiPlaneB2Model(num_classes=len(TARGET_FINDINGS)).to(device)

    if WEIGHTS_PATH and os.path.exists(WEIGHTS_PATH):
        print(f"Loading trained V5 weights from: {WEIGHTS_PATH}", flush=True)
        state_dict = torch.load(WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print("[SUCCESS] Loaded trained CoPAS B2 V5 model weights!", flush=True)
    else:
        print("[WARNING] Weights file not found!", flush=True)

    model.eval()
    predictions = []
    dummy_vol = torch.zeros((1, 16, 224, 224), dtype=torch.float32, device=device)

    with torch.no_grad():
        for i, row in test_df.iterrows():
            study_uid = row["StudyInstanceUID"]
            vols = extract_dual_contrast_slices(study_uid, series_df, num_slices=16)

            sag = torch.from_numpy(vols["Sagittal"]).unsqueeze(0).to(device) if "Sagittal" in vols else dummy_vol
            cor = torch.from_numpy(vols["Coronal"]).unsqueeze(0).to(device) if "Coronal" in vols else dummy_vol
            ax = torch.from_numpy(vols["Axial"]).unsqueeze(0).to(device) if "Axial" in vols else dummy_vol

            # Pass 1: Standard
            preds_orig = model(sag, cor, ax).squeeze().cpu().numpy()

            # Pass 2: TTA Horizontal Flip
            sag_flip = torch.flip(sag, dims=[-1])
            cor_flip = torch.flip(cor, dims=[-1])
            ax_flip = torch.flip(ax, dims=[-1])
            preds_flip = model(sag_flip, cor_flip, ax_flip).squeeze().cpu().numpy()

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

    print(f"\n[FINAL SUCCESS] Saved V5 submission to: {OUTPUT_SUB_CSV}", flush=True)
    print(sub_df.head())
    print("Checking NaNs:", sub_df.isna().sum().sum())
    print("Shape:", sub_df.shape)

if __name__ == "__main__":
    run_submission_v5()
