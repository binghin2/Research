# Biosignal-based Stress Classification

Deep learning pipeline for **stress / non-stress classification** using wearable biosignals (HR, HRV, EDA, ACC).  
Trained on **CATSA**, cross-dataset evaluated on **Empatica E4** — no subject overlap between train and test.

---

## Why Stress Classification?

Chronic stress is a key risk factor for cardiovascular disease, anxiety disorders, and burnout.  
Objective, continuous stress monitoring via everyday wearables enables early intervention.  
This project builds and benchmarks **10 deep learning architectures** on physiological time-series for real-world cross-dataset generalisation.

---

## Pipeline Overview

```
Raw biosignals (HR, HRV, EDA, ACC)
        │
        ▼
Per-subject normalisation  ←  Baseline-only (prevents label leakage)
        │
        ▼
Sliding window  (W = 60 s, stride = 10 s, 1 Hz)
        │
        ▼
Deep learning model  ──  Optuna HPO (15 trials × 2000 epochs)
        │
        ▼
Binary label: Non-stress (0)  /  Stress (1)
        │
        ▼
Cross-dataset evaluation on Empatica E4  →  F1 / Accuracy / AUROC
```

---

## Datasets

| Dataset | Subjects | Modalities | Tasks | Role |
|---------|----------|------------|-------|------|
| **CATSA** | 50 | HR, HRV, EDA, ACC | Baseline, Logic, N-back, Stroop, Sudoku | Train / Val |
| **Empatica E4** | 6 | HR, HRV | Rest / Cognitive stress | **Test only** |
| **WESAD** | 15 | HR, HRV, ACC, EDA | Baseline, Stress, Amusement | Supplementary |

> Datasets are **not included** due to size/licence constraints.  
> Access: [CATSA](https://www.nature.com/articles/s41597-024-03400-y) · [Empatica E4 Stress](https://www.kaggle.com/datasets/qiriro/stress) · [WESAD](https://uni-siegen.de/life/WESAD)

---

## Models Evaluated

| Model | Architecture |
|-------|-------------|
| **iTransformer** | Inverted Transformer |
| **Mamba2** | State Space Model (SSM) |
| **PatchTST** | Patch-based Transformer |
| **TimeMixerPP** | Multi-scale MLP Mixer |
| **Medformer** | Multi-granularity Transformer |
| **CrossGNN** | Cross-channel GNN |
| **TransformerCNN** | Hybrid CNN + Transformer |
| **TimesNet** | 2-D Temporal Convolution |
| **TSLANet** | Adaptive Spectral Layer Network |
| **ModernTCN** | Modern Temporal Convolutional Network |

---

## Results

### Signal: HR only — CATSA → Empatica E4

| Model | Val F1 | **E4 F1** | E4 Acc | E4 AUROC |
|-------|-------:|----------:|-------:|---------:|
| **Mamba2** | 0.907 | **0.954** | **0.915** | 0.705 |
| CrossGNN | 0.905 | 0.943 | 0.895 | 0.651 |
| PatchTST | 0.907 | 0.943 | 0.894 | 0.710 |
| Medformer | 0.897 | 0.942 | 0.890 | 0.631 |
| iTransformer | 0.899 | 0.937 | 0.885 | 0.659 |
| TransformerCNN | 0.909 | 0.937 | 0.884 | 0.642 |
| ModernTCN | 0.904 | 0.936 | 0.884 | 0.675 |
| TSLANet | 0.907 | 0.935 | 0.880 | 0.684 |
| TimesNet | 0.905 | 0.927 | 0.868 | 0.677 |
| TimeMixerPP | 0.901 | 0.914 | 0.846 | 0.666 |

### Signal: HR + HRV — Normalisation ablation (iTransformer)

| Normalisation | Val F1 | **E4 F1** | E4 Acc | E4 AUROC |
|--------------|-------:|----------:|-------:|---------:|
| All-task (original) | 0.898 | 0.823 | 0.707 | 0.625 |
| **Baseline-only (fixed)** | **0.911** | **0.926** | **0.866** | **0.833** |

> Baseline-only normalisation removes label leakage: **+10.3 pp F1**, **+18.1 pp AUROC** on E4.

### Signal: ALL (HR + HRV + EDA + ACC) — CATSA → Empatica E4

| Model | Val F1 | **E4 F1** | E4 Acc | E4 AUROC |
|-------|-------:|----------:|-------:|---------:|
| **Mamba2** | 0.970 | **0.950** | **0.906** | 0.675 |
| **TimeMixerPP** | 0.958 | **0.950** | **0.906** | **0.877** |
| CrossGNN | 0.976 | 0.944 | 0.899 | 0.735 |
| Medformer | 0.963 | 0.944 | 0.895 | 0.732 |
| iTransformer | 0.911 | 0.941 | 0.890 | 0.549 |
| TransformerCNN | 0.968 | 0.912 | 0.844 | 0.718 |
| TimesNet | 0.972 | 0.901 | 0.826 | 0.616 |
| TSLANet | 0.957 | 0.897 | 0.817 | 0.676 |
| PatchTST | 0.961 | 0.892 | 0.808 | 0.723 |
| ModernTCN | 0.964 | 0.877 | 0.787 | 0.643 |

---

## Project Structure

```
Research/
├── CATSA/
│   ├── Data_Analysis/               # EDA, signal visualisation notebooks
│   └── Train/
│       └── Individual_data/
│           ├── Only_HR/             # HR-only, 10 models, 2000 epochs
│           │   └── 2000epoch.ipynb
│           ├── HR&HRV/              # HR+HRV experiments
│           │   ├── HR_HRV_Norm.ipynb                    # baseline (all-task norm)
│           │   ├── HR_HRV_iTransformer_FixedNorm.ipynb  # fixed norm (paper version)
│           │   ├── HR&HRV_models_ten.py                 # 10 model definitions
│           │   └── HR_HRV_optuna_utils.py               # shared training utilities
│           ├── ALL/                 # All-signal (HR+HRV+EDA+ACC) experiments
│           │   └── ALL_Norm.ipynb
│           ├── EDA/                 # EDA-only experiments
│           └── ACC/                 # ACC-only experiments
├── WESAD_classification/            # WESAD dataset experiments
│   └── Training/
│       ├── Transformer/
│       ├── LSTM/
│       └── GRU/
├── Data_Preprocessing/              # Dataset preparation scripts
└── Pressuredata/                    # Wrist pressure sensor stress experiments
```

---

## Installation

```bash
git clone https://github.com/binghin2/Research.git
cd Research

conda create -n myenv python=3.10
conda activate myenv

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scipy scikit-learn matplotlib optuna jupyter
```

---

## Quick Start

1. Place datasets at the paths set in each notebook (`DATA_ROOT`, `E4_ROOT`).
2. Open the notebook of interest:
   - **Best cross-dataset model (HR only):** `CATSA/Train/Individual_data/Only_HR/2000epoch.ipynb`
   - **Normalisation-fixed iTransformer (HR+HRV):** `CATSA/Train/Individual_data/HR&HRV/HR_HRV_iTransformer_FixedNorm.ipynb`
   - **All-signal benchmark:** `CATSA/Train/Individual_data/ALL/ALL_Norm.ipynb`
3. Run all cells. Optuna studies are **resumable** via SQLite (`Save_model_*/optuna_studies.db`).

---

## Key Design Decisions

| Decision | Detail |
|----------|--------|
| **Normalisation** | Per-subject, baseline task only — prevents label leakage from stress tasks |
| **Val split** | Subject-level (not window-level) — no cross-subject contamination |
| **E4 normalisation** | Rest-segment only, matching CATSA Baseline rule |
| **Loss function** | `BCEWithLogitsLoss(pos_weight)` — handles Stress/Non-stress class imbalance |
| **Optimiser** | AdamW + cosine annealing LR decay + early stopping (patience 50–100) |
| **HPO** | Optuna TPE sampler, 15 trials, NopPruner, 2000 max epochs |

---

## Citation

If you use this code or results, please cite:

```bibtex
@misc{park2026biosignal,
  author    = {Byunghyun Mo},
  title     = {Biosignal-based Stress Classification with Deep Learning},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/binghin2/Research}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
