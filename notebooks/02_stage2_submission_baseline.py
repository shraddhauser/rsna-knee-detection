# ==============================================================================
# RSNA 2026 KNEE ABNORMALITY DETECTION: STAGE 2
# FAST DICOM INFERENCE & SUBMISSION BASELINE PIPELINE
# ==============================================================================
# Purpose:
# 1. Verifies end-to-end data pipeline on Kaggle test set without bugs or leaks.
# 2. Safely reads DICOMs with windowing, selects central 16 slices per plane.
# 3. Outputs valid submission.csv matching sample_submission.csv format.
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

# Path Auto-Discovery
DATA_DIR = None
if os.path.exists("/kaggle/input"):
    for root, dirs, files in os.walk("/kaggle/input"):
        if "sample_submission.csv" in files:
            DATA_DIR = root
            break
if not DATA_DIR:
    DATA_DIR = "/kaggle/input/rsna-knee-abnormality-detection"

print(f"Data directory: {DATA_DIR}")

TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TEST_SERIES_CSV = os.path.join(DATA_DIR, "test_series.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
TEST_SERIES_DIR = os.path.join(DATA_DIR, "test_series")
OUTPUT_SUB_CSV = "/kaggle/working/submission.csv"

TARGET_FINDINGS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

def read_dicom_slice(filepath: str, target_size=(224, 224)) -> Optional[np.ndarray]:
    """Read single DICOM slice safely and normalize to float32 [0, 1]."""
    try:
        dcm = pydicom.dcmread(filepath, force=True)
        img = dcm.pixel_array.astype(np.float32)
        
        slope = float(getattr(dcm, "RescaleSlope", 1.0))
        intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
        img = img * slope + intercept
        
        p_low, p_high = np.percentile(img, (1, 99))
        if p_high > p_low:
            img = np.clip(img, p_low, p_high)
            img = (img - p_low) / (p_high - p_low)
        else:
            img = np.zeros_like(img)

        t_img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
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

class KneeMultiPlaneModel(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        orig_conv = backbone.features[0][0]
        self.encoder = backbone.features
        self.encoder[0][0] = nn.Conv2d(
            16, orig_conv.out_channels, 
            kernel_size=orig_conv.kernel_size, 
            stride=orig_conv.stride, 
            padding=orig_conv.padding, 
            bias=False
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.pool(feat).flatten(1)
        return self.classifier(feat)

def generate_submission():
    print("Loading test metadata...")
    test_df = pd.read_csv(TEST_CSV)
    series_df = pd.read_csv(TEST_SERIES_CSV)
    print(f"Total test studies: {len(test_df)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = KneeMultiPlaneModel(num_classes=len(TARGET_FINDINGS)).float().to(device)
    model.eval()

    baseline_priors = {
        "ACL": 0.28, "MCL": 0.16, "Medial Meniscus": 0.38, "Lateral Meniscus": 0.21,
        "Medial OA": 0.32, "Lateral OA": 0.18, "PF OA": 0.26, "Effusion": 0.44,
        "Synovitis": 0.19, "Baker's": 0.15, "Contusion": 0.17, "Fracture": 0.08
    }

    predictions = []
    with torch.no_grad():
        for i, row in test_df.iterrows():
            study_uid = row["StudyInstanceUID"]
            volumes = extract_study_slices(study_uid, series_df, num_slices=16)
            pred_scores = {f: baseline_priors[f] for f in TARGET_FINDINGS}

            if volumes:
                chosen_plane = "Sagittal" if "Sagittal" in volumes else list(volumes.keys())[0]
                vol_tensor = torch.from_numpy(volumes[chosen_plane]).unsqueeze(0).float().to(device)
                preds = model(vol_tensor).squeeze().cpu().numpy()

                for idx, f in enumerate(TARGET_FINDINGS):
                    pred_scores[f] = float(0.5 * preds[idx] + 0.5 * baseline_priors[f])

            pred_scores["StudyInstanceUID"] = study_uid
            predictions.append(pred_scores)

            if (i + 1) % 50 == 0 or (i + 1) == len(test_df):
                print(f"Processed {i + 1}/{len(test_df)} studies")

    sub_df = pd.DataFrame(predictions)
    cols = ["StudyInstanceUID"] + TARGET_FINDINGS
    sub_df = sub_df[cols]

    sub_df.to_csv(OUTPUT_SUB_CSV, index=False)
    print(f"\n[SUCCESS] Generated submission at: {OUTPUT_SUB_CSV}")
    print(sub_df.head())
    print("\nSubmission shape:", sub_df.shape)
    print("Checking for NaNs:", sub_df.isna().sum().sum())

if __name__ == "__main__":
    generate_submission()
