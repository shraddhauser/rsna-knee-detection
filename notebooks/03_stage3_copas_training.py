# ==============================================================================
# RSNA 2026 KNEE ABNORMALITY DETECTION: FAST 15-MIN STAGE 3
# CURATED HIGH-SIGNAL CoPAS TRAINING (GPU ACCELERATED)
# ==============================================================================
import os
import glob
import math
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from typing import Dict, List, Optional

# 1. Path Setup
DATA_DIR = None
if os.path.exists("/kaggle/input"):
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train_series.csv" in files:
            DATA_DIR = root
            break
if not DATA_DIR:
    DATA_DIR = "/kaggle/input/rsna-knee-abnormality-detection"

print(f"Data directory: {DATA_DIR}")

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_SERIES_CSV = os.path.join(DATA_DIR, "train_series.csv")
TRAIN_SERIES_DIR = os.path.join(DATA_DIR, "train_series")
OUTPUT_MODEL_PATH = "/kaggle/working/best_knee_model.pth"

TARGET_FINDINGS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

# 2. Fast DICOM Loader
def read_dicom_slice(filepath: str, target_size=(224, 224)) -> Optional[np.ndarray]:
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

        series_path = os.path.join(TRAIN_SERIES_DIR, study_uid, chosen_series_uid)
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

# 3. Smart Dataset Curator (Filters to top 600 informative studies)
def curate_informative_dataset(train_df: pd.DataFrame, max_samples: int = 600) -> pd.DataFrame:
    print("Curating high-yield informative study subset...")
    for f in TARGET_FINDINGS:
        if f not in train_df.columns:
            train_df[f] = np.nan
        train_df[f + "_target"] = train_df[f].fillna(0.20).astype(np.float32)

    gold_mask = train_df[TARGET_FINDINGS].notna().any(axis=1)
    gold_df = train_df[gold_mask]
    remaining_df = train_df[~gold_mask]

    n_needed = max_samples - len(gold_df)
    sampled_remaining = remaining_df.sample(n=min(n_needed, len(remaining_df)), random_state=42)

    curated = pd.concat([gold_df, sampled_remaining]).reset_index(drop=True)
    print(f"Curated {len(curated)} high-signal studies ({len(gold_df)} gold + {len(sampled_remaining)} report-derived).")
    return curated

# 4. Dataset
class KneeMRIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, series_df: pd.DataFrame, num_slices=16):
        self.df = df.reset_index(drop=True)
        self.series_df = series_df
        self.num_slices = num_slices
        self.targets = self.df[[f + "_target" for f in TARGET_FINDINGS]].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        study_uid = self.df.iloc[idx]["StudyInstanceUID"]
        volumes = extract_study_slices(study_uid, self.series_df, self.num_slices)
        target = self.targets[idx]

        dummy = np.zeros((self.num_slices, 224, 224), dtype=np.float32)
        sag = volumes.get("Sagittal", dummy)
        cor = volumes.get("Coronal", dummy)
        ax = volumes.get("Axial", dummy)

        return (
            torch.from_numpy(sag).float(),
            torch.from_numpy(cor).float(),
            torch.from_numpy(ax).float(),
            torch.from_numpy(target).float()
        )

# 5. CoPAS Architecture
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
        return logits

# 6. Combined Loss
class CombinedFocalBCELoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        focal_weight = self.alpha * (1 - pt).pow(self.gamma)
        return (focal_weight * bce).mean()

# 7. Fast Train Loop
def run_fast_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_df = pd.read_csv(TRAIN_CSV)
    series_df = pd.read_csv(TRAIN_SERIES_CSV)
    curated_df = curate_informative_dataset(train_df, max_samples=600)

    val_df = curated_df.iloc[:60]
    train_split = curated_df.iloc[60:]

    train_loader = DataLoader(KneeMRIDataset(train_split, series_df), batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(KneeMRIDataset(val_df, series_df), batch_size=8, shuffle=False, num_workers=2)

    model = CoPASMultiPlaneModel(num_classes=len(TARGET_FINDINGS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = CombinedFocalBCELoss()
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    EPOCHS = 3
    print(f"\nStarting 3-epoch fast training loop ({len(train_split)} train, {len(val_df)} val)...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for i, (sag, cor, ax, targets) in enumerate(train_loader):
            sag, cor, ax, targets = sag.to(device), cor.to(device), ax.to(device), targets.to(device)
            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast('cuda'):
                    logits = model(sag, cor, ax)
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(sag, cor, ax)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            if (i + 1) % 15 == 0:
                print(f"Epoch [{epoch}/{EPOCHS}] Step [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
        print(f"=== Epoch {epoch} Complete | Model checkpoint updated -> {OUTPUT_MODEL_PATH} ===")

    print("\n[SUCCESS] Model training complete and saved to:", OUTPUT_MODEL_PATH)

if __name__ == "__main__":
    run_fast_train()
