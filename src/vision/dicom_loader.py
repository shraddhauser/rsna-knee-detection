"""
Vision Module: DICOM Loader and Multi-Plane Slab Extractor
RSNA 2026 Knee Abnormality Detection AI Challenge
"""

import os
import glob
import numpy as np
import pydicom
import torch
import torch.nn as nn
from typing import Dict, Optional

def read_dicom_slice(filepath: str, target_size=(224, 224)) -> Optional[np.ndarray]:
    """Read a single DICOM slice with rescale slope/intercept and percentile windowing."""
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

def extract_study_slices(study_uid: str, series_df, base_series_dir: str, num_slices: int = 16) -> Dict[str, np.ndarray]:
    """Extract central 16-slice slabs for Sagittal, Coronal, and Axial planes."""
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

        series_path = os.path.join(base_series_dir, study_uid, chosen_series_uid)
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
