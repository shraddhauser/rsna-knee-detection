# RSNA 2026 Knee Abnormality Detection AI Challenge
## Project Progress & Checkpoint Summary

### Current Status: 0.584 Achieved (Version 9), Stage 4 V4 Deployed
- **Leaderboard Progression**:
  - `Version 3/5`: **0.496 – 0.497** (Initial constant-prediction baseline)
  - `Version 7`: **0.533** (Stage 4 V3 proof-of-concept on 630 CPU studies + Astuto cropper + TTA)
  - `Version 9`: **0.584** (+0.051 leap! GPU-trained CoPAS weights on 2,000 studies)
- **Active Step**: Deploying **Stage 4 V4** (`04_stage4_final_submission_v4.py`) on Kaggle:
  - Multi-Series Ensembling: fuses both Fluid-Sensitive (T2/STIR) and Anatomical (T1/PD) series configurations per patient.
  - 3-Way Test-Time Augmentation (TTA): Original + Horizontal Mirror Flip + Contrast Invariant passes.
  - Direct Path Loading: Instant startup bypassing slow recursive directory scans.
  - Zero GPU Training Quota burned: Preserves remaining ~4 hours of weekly GPU quota.

---

### Project File Structure

```
c:\Users\Nitin\Desktop\Rsna\
├── notebooks/
│   ├── 01_stage1_nlp_weak_labels.py          # Stage 1: Multilingual NegEx report extractor
│   ├── 02_stage2_submission_baseline.py      # Stage 2: Submission validator
│   ├── 03_stage3_copas_training_v4.py        # Stage 3 V4: High-power 2,000-study GPU trainer
│   ├── 03_stage3_copas_training_v5.py        # Stage 3 V5: Precompiled fast NLP + B2 trainer
│   ├── 04_stage4_final_submission_v3.py      # Stage 4 V3: Scored 0.584 on LB
│   └── 04_stage4_final_submission_v4.py      # Stage 4 V4: Multi-series fusion + 3-way TTA
└── src/
    ├── nlp/weak_label_engine.py              # Multilingual NegEx NLP (precompiled regexes)
    ├── vision/knee_cropper.py                # Astuto et al. ROI joint cropper
    ├── vision/dicom_loader.py                # 2.5D percentile windowing & slice sampler
    ├── models/copas_network.py               # CoPAS multi-plane attention network
    └── training/loss_functions.py            # Asymmetric Focal + BCE loss
```
