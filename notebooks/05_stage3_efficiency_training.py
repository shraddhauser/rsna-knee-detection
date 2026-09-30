# ==============================================================================
# RSNA 2026 KNEE ABNORMALITY DETECTION: EFFICIENCY-OPTIMIZED TRAINING
# PRE-CACHED I/O + LIGHTWEIGHT CoPAS + 8-SLICE FAST ARCHITECTURE
# ==============================================================================
# Strategy: Separate slow DICOM I/O from fast GPU training
# Phase 1: Pre-cache 1,000 studies into RAM (~8 min)
# Phase 2: Train 5 epochs from cached data (~5 min on GPU)
# Total: ~15 minutes
# ==============================================================================

import os, re, glob, unicodedata, time
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from typing import Dict, List, Optional

t_start = time.time()

# 1. Direct Paths (NO os.walk - instant)
COMP_DIR = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
if not os.path.exists(COMP_DIR):
    COMP_DIR = "/kaggle/input/rsna-knee-abnormality-detection"
print(f"Data directory: {COMP_DIR}", flush=True)

TRAIN_CSV = os.path.join(COMP_DIR, "train.csv")
TRAIN_SERIES_CSV = os.path.join(COMP_DIR, "train_series.csv")
TRAIN_SERIES_DIR = os.path.join(COMP_DIR, "train_series")
OUTPUT_MODEL_PATH = "/kaggle/working/best_knee_model.pth"
NUM_SLICES = 8  # Efficiency: 8 slices instead of 16 (2x faster I/O + inference)

TARGET_FINDINGS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

# 2. Fast Precompiled NegEx NLP Parser
def normalize_text(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return re.sub(r'\s+', ' ', text.lower()).strip()

TARGET_PATTERNS = {
    "ACL": [r"\banterior cruciate ligament\b", r"\bacl\b", r"\bligamento cruzado anterior\b", r"\blca\b"],
    "MCL": [r"\bmedial collateral ligament\b", r"\bmcl\b", r"\bligamento colateral medial\b"],
    "Medial Meniscus": [r"\bmedial meniscus\b", r"\bmedial meniscal\b", r"\bmenisco medial\b"],
    "Lateral Meniscus": [r"\blateral meniscus\b", r"\blateral meniscal\b", r"\bmenisco lateral\b"],
    "Medial OA": [r"\bmedial\b.*?\b(?:osteoarthritis|oa|arthrosis|chondromalacia|cartilage loss)\b"],
    "Lateral OA": [r"\blateral\b.*?\b(?:osteoarthritis|oa|arthrosis|chondromalacia|cartilage loss)\b"],
    "PF OA": [r"\bpatellofemoral\b.*?\b(?:osteoarthritis|oa|arthrosis|chondromalacia)\b", r"\bretropatellar\b"],
    "Effusion": [r"\beffusion\b", r"\bderrame\b", r"\berguss\b"],
    "Synovitis": [r"\bsynovitis\b", r"\bsynovial\b", r"\bsinovitis\b"],
    "Baker's": [r"\bbaker(?:'s)? cyst\b", r"\bpopliteal cyst\b"],
    "Contusion": [r"\bbone contusion\b", r"\bbone bruise\b", r"\bmarrow edema\b"],
    "Fracture": [r"\bfracture\b", r"\bfractured\b", r"\bfractura\b", r"\bfraktur\b"]
}

PRE_NEG = [r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\bnegative for\b", r"\bunremarkable\b", r"\bnormal\b", r"\bpreserved\b", r"\babsence\b", r"\bkein\b", r"\bohne\b"]
POST_NEG = [r"\bunremarkable\b", r"\bintact\b", r"\bnormal\b", r"\bnot seen\b"]

class FastReportParser:
    def __init__(self):
        self.pre_re = re.compile(r"|".join(PRE_NEG), re.IGNORECASE)
        self.post_re = re.compile(r"|".join(POST_NEG), re.IGNORECASE)
        self.compiled = {f: [re.compile(p, re.IGNORECASE) for p in pats] for f, pats in TARGET_PATTERNS.items()}

    def parse(self, text):
        norm = normalize_text(text)
        if not norm: return {f: 0.02 for f in TARGET_FINDINGS}
        clauses = [c.strip() for c in re.split(r'[.;:!?\n]+', norm) if len(c.strip()) > 2]
        scores = {f: 0.02 for f in TARGET_FINDINGS}
        for clause in clauses:
            for f in TARGET_FINDINGS:
                for pat in self.compiled[f]:
                    m = pat.search(clause)
                    if m:
                        pre = clause[max(0, m.start()-40):m.start()]
                        suf = clause[m.end():min(len(clause), m.end()+40)]
                        is_neg = bool(self.pre_re.search(pre) or self.post_re.search(suf))
                        scores[f] = 0.02 if is_neg else max(scores[f], 0.95)
                        break
        return scores

# 3. Efficient DICOM Reader (No Knee Cropper During Training for Speed)
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
        # Fast center crop instead of full Astuto (saves ~30% per slice)
        h, w = img.shape
        ch, cw = h // 2, w // 2
        margin = min(h, w) * 4 // 10
        img = img[max(0,ch-margin):ch+margin, max(0,cw-margin):cw+margin]
        t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        t = nn.functional.interpolate(t, size=target_size, mode="bilinear", align_corners=False)
        return t.squeeze().numpy().astype(np.float32)
    except Exception:
        return None

def load_study_volume(study_uid, series_df, num_slices=8):
    study_series = series_df[series_df["StudyInstanceUID"] == study_uid]
    volumes = {}
    for plane in ["Sagittal", "Coronal", "Axial"]:
        matched = study_series[study_series["Anatomical_Plane"] == plane]
        if matched.empty: continue
        if "Fluid_Sensitive" in matched.columns and (matched["Fluid_Sensitive"] == 1).any():
            s_uid = matched[matched["Fluid_Sensitive"] == 1].iloc[0]["SeriesInstanceUID"]
        else:
            s_uid = matched.iloc[0]["SeriesInstanceUID"]
        s_path = os.path.join(TRAIN_SERIES_DIR, study_uid, s_uid)
        if not os.path.exists(s_path): continue
        dcm_files = sorted(glob.glob(os.path.join(s_path, "*.dcm")))
        if len(dcm_files) == 0: continue
        indices = np.linspace(0, len(dcm_files)-1, num_slices, dtype=int)
        slices = []
        for idx in indices:
            sl = read_dicom_fast(dcm_files[idx])
            if sl is not None: slices.append(sl)
        if len(slices) >= num_slices:
            volumes[plane] = np.stack(slices[:num_slices], axis=0).astype(np.float32)
        elif len(slices) > 0:
            reps = int(np.ceil(num_slices / len(slices)))
            volumes[plane] = np.stack((slices * reps)[:num_slices], axis=0).astype(np.float32)
    return volumes

# ============================================================
# PHASE 1: BUILD TARGETS + PRE-CACHE ALL DATA TO RAM
# ============================================================
print("=" * 60, flush=True)
print("PHASE 1: Building targets & pre-caching DICOM data to RAM...", flush=True)
print("=" * 60, flush=True)

train_df = pd.read_csv(TRAIN_CSV)
series_df = pd.read_csv(TRAIN_SERIES_CSV)

# Build targets from reports
parser = FastReportParser()
target_data = []
for _, row in train_df.iterrows():
    report = row.get("Report", "")
    extracted = parser.parse(str(report)) if pd.notna(report) else {}
    row_targets = {}
    for f in TARGET_FINDINGS:
        gold = row.get(f)
        row_targets[f] = float(gold) if pd.notna(gold) else extracted.get(f, 0.02)
    target_data.append(row_targets)

target_df = pd.DataFrame(target_data)
for col in target_df.columns:
    train_df[col + "_t"] = target_df[col].values

# Balance dataset
tcols = [f + "_t" for f in TARGET_FINDINGS]
has_pos = (train_df[tcols] > 0.5).any(axis=1)
positives = train_df[has_pos]
negatives = train_df[~has_pos]
n_pos = min(len(positives), 500)
n_neg = min(len(negatives), 500)
curated = pd.concat([positives.sample(n_pos, random_state=42), negatives.sample(n_neg, random_state=42)]).reset_index(drop=True)

print(f"Cohort: {len(curated)} studies ({n_pos} positive, {n_neg} controls)", flush=True)
for f in TARGET_FINDINGS:
    r = (curated[f + "_t"] > 0.5).mean()
    print(f"  {f:18s}: {r*100:.1f}%", flush=True)

# Pre-cache all volumes to RAM
dummy = np.zeros((NUM_SLICES, 224, 224), dtype=np.float32)
cached_sag, cached_cor, cached_ax, cached_targets = [], [], [], []

for i, (_, row) in enumerate(curated.iterrows()):
    uid = row["StudyInstanceUID"]
    vols = load_study_volume(uid, series_df, NUM_SLICES)
    cached_sag.append(vols.get("Sagittal", dummy))
    cached_cor.append(vols.get("Coronal", dummy))
    cached_ax.append(vols.get("Axial", dummy))
    cached_targets.append(np.array([row[f + "_t"] for f in TARGET_FINDINGS], dtype=np.float32))
    if (i + 1) % 50 == 0 or (i + 1) == len(curated):
        elapsed = time.time() - t_start
        print(f"  Cached {i+1}/{len(curated)} studies ({elapsed:.0f}s elapsed)", flush=True)

cached_sag = np.stack(cached_sag)
cached_cor = np.stack(cached_cor)
cached_ax = np.stack(cached_ax)
cached_targets = np.stack(cached_targets)
print(f"\nPhase 1 complete! Cached shape: {cached_sag.shape} ({time.time()-t_start:.0f}s total)", flush=True)

# ============================================================
# PHASE 2: FAST GPU TRAINING FROM CACHED DATA
# ============================================================
print("\n" + "=" * 60, flush=True)
print("PHASE 2: Training from cached data (pure GPU speed)...", flush=True)
print("=" * 60, flush=True)

class CachedDataset(Dataset):
    def __init__(self, sag, cor, ax, targets, is_train=True):
        self.sag, self.cor, self.ax = sag, cor, ax
        self.targets = targets
        self.is_train = is_train

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        s, c, a, t = self.sag[idx].copy(), self.cor[idx].copy(), self.ax[idx].copy(), self.targets[idx]
        if self.is_train:
            if np.random.rand() > 0.5:
                s = np.flip(s, axis=-1).copy()
                c = np.flip(c, axis=-1).copy()
                a = np.flip(a, axis=-1).copy()
            j = np.random.uniform(0.9, 1.1)
            s = np.clip(s * j, 0, 1)
            c = np.clip(c * j, 0, 1)
            a = np.clip(a * j, 0, 1)
        return torch.from_numpy(s).float(), torch.from_numpy(c).float(), torch.from_numpy(a).float(), torch.from_numpy(t).float()

class CoPASEfficient(nn.Module):
    def __init__(self, num_classes=12, num_slices=8):
        super().__init__()
        try:
            base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            print("[INFO] ImageNet pretrained weights loaded.", flush=True)
        except:
            base = models.efficientnet_b0(weights=None)

        orig = base.features[0][0]
        new_conv = nn.Conv2d(num_slices, orig.out_channels, kernel_size=orig.kernel_size, stride=orig.stride, padding=orig.padding, bias=False)
        with torch.no_grad():
            w = orig.weight.repeat(1, (num_slices//3)+1, 1, 1)[:, :num_slices, :, :]
            new_conv.weight.copy_(w / (num_slices / 3.0))
        base.features[0][0] = new_conv

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
        return self.head(fused)

# Split
n_val = 80
val_ds = CachedDataset(cached_sag[:n_val], cached_cor[:n_val], cached_ax[:n_val], cached_targets[:n_val], is_train=False)
train_ds = CachedDataset(cached_sag[n_val:], cached_cor[n_val:], cached_ax[n_val:], cached_targets[n_val:], is_train=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training device: {device}", flush=True)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

model = CoPASEfficient(num_classes=12, num_slices=NUM_SLICES).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
EPOCHS = 5
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
bce = nn.BCEWithLogitsLoss()
scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

best_val = float("inf")
for epoch in range(1, EPOCHS + 1):
    model.train()
    tloss = 0.0
    for i, (s, c, a, t) in enumerate(train_loader):
        s, c, a, t = s.to(device), c.to(device), a.to(device), t.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.amp.autocast('cuda'):
                loss = bce(model(s, c, a), t)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = bce(model(s, c, a), t)
            loss.backward()
            optimizer.step()
        tloss += loss.item()
    scheduler.step()

    model.eval()
    vloss = 0.0
    with torch.no_grad():
        for s, c, a, t in val_loader:
            s, c, a, t = s.to(device), c.to(device), a.to(device), t.to(device)
            if scaler:
                with torch.amp.autocast('cuda'):
                    loss = bce(model(s, c, a), t)
            else:
                loss = bce(model(s, c, a), t)
            vloss += loss.item()

    tl = tloss / len(train_loader)
    vl = vloss / max(1, len(val_loader))
    elapsed = time.time() - t_start
    print(f"Epoch [{epoch}/{EPOCHS}] Train: {tl:.4f} | Val: {vl:.4f} ({elapsed:.0f}s)", flush=True)

    if vl < best_val:
        best_val = vl
        torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
        sz = os.path.getsize(OUTPUT_MODEL_PATH) / 1024 / 1024
        print(f"  [*] Saved best model ({sz:.1f} MB, Val: {best_val:.4f})", flush=True)

total = time.time() - t_start
print(f"\n[SUCCESS] Training completed in {total:.0f} seconds ({total/60:.1f} minutes)!", flush=True)
print(f"Model saved to: {OUTPUT_MODEL_PATH}", flush=True)
