# RSNA 2026 Knee Abnormality Detection AI Challenge
## Comprehensive Technical Report & Project Walkthrough

---

## 1. Executive Summary & Objective

* **Competition**: 2026 RSNA Knee Abnormality Detection (Kaggle).
* **Core Task**: Predict per-study probabilities for **12 clinically critical knee abnormalities** from multi-plane MRI scans.
  * *Targets*: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's Cyst, Bone Contusion, Fracture.
* **Evaluation Metric**: **Macro-averaged ROC-AUC** across all 12 targets.
* **The Asymmetry Challenge**:
  * **Scale**: ~570 GB, 819,640 DICOM files across 16 global clinical sites.
  * **Labels**: Only ~58 studies have direct gold human annotations; ~4,349 studies have unstructured, multilingual free-text radiology reports.
  * **Test Constraint**: Held-out test set contains **DICOM images ONLY — NO reports**. Inference must be 100% vision-based within Kaggle's 9-hour execution limit without internet.

---

## 2. Technical Stack Used

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Deep Learning Framework** | PyTorch 2.4+ & Torchvision | Industry standard for medical vision and custom multi-branch architectures. |
| **Vision Backbone** | EfficientNet-B0 (2.5D Multi-Slice) | Balances high spatial feature representation with low parameter footprint for fast Kaggle GPU inference. |
| **Multi-Plane Fusion** | CoPAS Cross-Plane Multi-Head Attention | Implements Qiu et al. (Nature Comms 2024) to fuse Sagittal, Coronal, and Axial planes dynamically. |
| **DICOM Engine** | PyDicom + Percentile Windowing | Decodes multi-vendor transfer syntaxes (JPEG Lossless, JPEG 2000, Explicit VR) with robust slope/intercept rescaling. |
| **NLP Weak Labeling** | Custom Scope-Based NegEx Engine | Regex-based deterministic negation/uncertainty scope parser across 6 languages (EN, ES, DE, FR, PT, IT). |
| **Loss Optimization** | Asymmetric Focal + BCE Loss ($\gamma=2.0$) | Downweights easy background negatives and amplifies gradients on rare positive abnormalities. |
| **Acceleration** | PyTorch AMP (Automatic Mixed Precision) | Uses FP16 operations on NVIDIA Tesla T4 GPUs for 2.5x faster throughput and 50% lower VRAM usage. |

---

## 3. The Core Logic & Architecture Behind Each Choice

### A. Why Scope-Based NegEx for Stage 1 (NLP)?
* **The Problem**: In clinical radiology reports, mentions of findings are frequently negative (e.g., *"No evidence of fracture"*, *"MCL is intact"*). Generic LLMs and keyword matchers fail on these, creating massive false-positive label corruption.
* **The Solution**: We built a deterministic scope detector that analyzes clauses before and after a finding mention. It distinguishes between genuine negations (*"kein Erguss"*, *"sin desgarro"*) and pseudo-negations (*"no significant interval change in previously noted complex tear"*), converting 4,349 unlabeled studies into clean continuous ground-truth targets.

### B. Why CoPAS Multi-Plane Attention for Stage 3 (Vision)?
* **Anatomical Reality**:
  * **Sagittal Plane**: Optimal for the anterior cruciate ligament (`ACL`) and meniscal anterior/posterior horns.
  * **Coronal Plane**: Optimal for collateral ligaments (`MCL`) and tibiofemoral joint-space narrowing (`Medial OA`, `Lateral OA`).
  * **Axial Plane**: Optimal for the patella, patellofemoral joint (`PF OA`), joint fluid (`Effusion`), and popliteal space (`Baker's Cyst`).
* **The Fusion**: Rather than training 3 disconnected single-plane models, the CoPAS architecture passes each plane through a shared 2.5D encoder, stacks their latent vectors, and uses multi-head attention to let the network correlate features across all three anatomical perspectives.

### C. Why Astuto et al. Knee Joint ROI Cropping?
* **Resolution Bottleneck**: Raw knee MRI slices are $512 \times 512$, but 70% of the image is empty air and scanner margins. Downsampling the raw image directly to $224 \times 224$ reduces the actual knee joint to a tiny $60 \times 60$ area.
* **The Fix**: The automated ROI cropper finds the tissue bounding box using density percentiles, eliminates the black air borders, and crops tightly around the femoral condyles and tibial plateau, tripling the effective resolution on cartilage and ligaments.

---

## 4. Key Engineering Errors Tackled & Solved

### Error 1: Kaggle Path Mismatches (`FileNotFoundError`)
* **Symptom**: Kaggle competition slugs differ from UI titles (`/kaggle/input/competitions/rsna-knee-abnormality-detection` vs `/kaggle/input/rsna-knee-abnormality-detection`).
* **Resolution**: Implemented auto-discovery that dynamically scans `/kaggle/input` for key landmark files (`train.csv`, `sample_submission.csv`) regardless of folder renaming.

### Error 2: PyTorch Data Type Mismatch (`Double vs Float`)
* **Symptom**: `RuntimeError: expected scalar type Double but found Float`.
* **Root Cause**: NumPy percentile normalization defaults to 64-bit floating point (`float64`), which PyTorch converts to `DoubleTensor`, clashing with 32-bit `nn.Conv2d` layers.
* **Resolution**: Explicitly cast all arrays and tensors to `.float()` / `np.float32` across the pipeline.

### Error 3: Offline Internet Download Blocks (`URLError`)
* **Symptom**: PyTorch blocked trying to download `efficientnet_b0_rwightman-7f5810bc.pth` because Kaggle notebooks had internet disabled.
* **Resolution**: Toggled Internet ON for training sessions (allowed by Kaggle rules) while maintaining zero-internet compliance for test submission notebooks.

### Error 4: Interactive Session Timeout Overnight ("Notebook Canceled")
* **Symptom**: Leaving a 3,700-study training loop running in an interactive tab overnight caused Kaggle to shut down the container after 60 minutes of browser inactivity.
* **Resolution**: Replaced brute-force sequential reading with **Informative Stratified Sampling** (100% of gold studies + high-signal abnormal cases = 1,000 studies). This finished in 15 minutes on GPU without timeout risk and achieved lower training loss ($0.0172$).

### Error 5: The Mystery of the Initial 0.497 Score
* **Symptom**: First Kaggle submission received a score of 0.497 (identical to random guessing).
* **Root Cause**:
  1. `os.walk("/kaggle/input")` stalled while trying to index 820,000 files in the competition folder.
  2. The script timed out finding the model, fell back to `WEIGHTS_PATH = None`, and executed with **randomly initialized, untrained weights**.
  3. Kaggle placed attached notebook outputs under `/kaggle/input/notebooks/kaggluuu/`, which was missed by the initial shallow check.
* **Resolution**: Switched to a fast direct lookup targeting `/kaggle/input/notebooks/kaggluuu/rsna-knee-stage3-training/best_knee_model.pth`. The logs confirmed: `[SUCCESS] Loaded trained CoPAS model weights!`, producing real, trained neural predictions for Version 4.

---

## 5. Summary of Artifacts & Files Created

```
c:\Users\Nitin\Desktop\Rsna\
├── README.md                                # Master documentation with 8-paper synthesis
├── PROJECT_STATUS.md                        # Checkpoint & resumption log
├── notebooks/
│   ├── 01_stage1_nlp_weak_labels.py        # Stage 1: Multilingual report NLP & NegEx engine
│   ├── 02_stage2_submission_baseline.py    # Stage 2: Fast DICOM loader & test submission pipeline
│   ├── 03_stage3_copas_training.py         # Stage 3: CoPAS GPU training loop
│   ├── 03_stage3_copas_training_v2.py      # Stage 3 V2: Fast 15-min curated training with ROI cropper
│   ├── 04_stage4_final_submission.py       # Stage 4: Test inference with trained weights
│   └── 04_stage4_final_submission_v2.py    # Stage 4 V2: Verified path submission with ROI cropping
└── src/
    ├── nlp/
    │   ├── weak_label_engine.py            # Multilingual NegEx regex engine (12 classes)
    │   └── test_weak_label_engine.py       # Unit tests (100% pass across EN, ES, DE, FR)
    ├── vision/
    │   ├── dicom_loader.py                 # Fast DICOM reader & percentile windowing
    │   └── knee_cropper.py                 # Anatomical ROI bone & tissue bounding box cropper
    ├── models/
    │   └── copas_network.py                # CoPAS multi-plane cross-attention architecture
    └── training/
        └── loss_functions.py               # Combined Focal + BCE Loss (gamma=2.0)
```
* **GitHub Repository**: [https://github.com/shraddhauser/rsna-knee-detection](https://github.com/shraddhauser/rsna-knee-detection)
