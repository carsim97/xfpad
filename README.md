# X-FPAD: Fingerprint PAD Exposimeter

Reference implementation for the paper

> **The Fingerprint PAD Exposimeter (X-FPAD): A Visual Framework for Evaluating Generalization to Unseen Attacks.**
> Simone Carta, Roberto Casula, Gian Luca Marcialis. University of Cagliari.

X-FPAD reformulates fingerprint Presentation Attack Detection (PAD) as a structured-manifold learning problem. A frozen MobileNet-v2 backbone produces 1280-D fingerprint embeddings; a lightweight MLP (the geometric encoder) projects them into a deterministic 2-D latent space governed by a radial-angular geometry — bona fide samples are anchored at the origin, each known PAI occupies a dedicated angular sector, and radial distance encodes deviation from authentic skin. The induced manifold is used as a diagnostic surface: unseen PAIs are projected onto it and their dominant angular anchors are identified by direct inspection or via the soft-attribution metric `p_{u,k}`. Targeted ablations on three CNN backbones (MobileNet-v2, ResNet-18, DenseNet-121) confirm that the directional dependencies exposed by X-FPAD are intrinsic feature-space properties rather than backbone-specific artefacts.

This repository contains everything needed to reproduce the experimental pipeline of the paper end-to-end.

---

## Repository structure

```
xfpad-code/
├── configs/
│   ├── base.yaml                 # shared hyperparameters
│   ├── greenbit.yaml             # Green Bit DactyScan84C
│   └── dermalog.yaml             # Dermalog LF10
├── xfpad/                        # importable package
│   ├── config.py                 # YAML loader with deep merge
│   ├── utils.py                  # seed/device/path helpers
│   ├── data/                     # datasets, label rules, MINDTCT patches
│   ├── models/                   # FeatureExtractor, GeometricEncoder, PadDetector
│   ├── losses/                   # ConcentricLoss, AngularLoss + log-scaling
│   ├── metrics/                  # BFO/RCI/ACS, p_{u,k}, APCER/BPCER
│   ├── training/                 # circular ordering of PAI prototypes
│   └── viz/                      # 2-D latent-space plotting
├── scripts/                      # CLI entry points
│   ├── extract_patches.py        # 224x224 minutiae-guided patches
│   ├── optimize_ordering.py      # cosine-similarity-based label assignment
│   ├── phase1_train.py           # backbone + features + geometric encoder
│   ├── phase2_map_unseen.py      # directional mapping of unseen PAIs
│   └── phase3_audit_pad.py       # binary PAD ablation experiments
├── data/
│   ├── splits/                   # *.txt files (one image path per line)
│   └── features/                 # cached 1280-D embeddings (.npy)
├── checkpoints/                  # model checkpoints written by the scripts
├── outputs/                      # plots, JSON tables, projection caches
├── requirements.txt
├── setup.py
└── README.md
```

---

## Installation

Tested with Python 3.9.

```bash
git clone https://github.com/<your-org>/xfpad.git
cd xfpad
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .
```

The package installs `torch`, `torchvision`, `numpy`, `opencv-python`, `Pillow`, `matplotlib`, `tqdm`, `PyYAML`, and `scikit-learn` (versions pinned in `requirements.txt`).

A CUDA-enabled GPU is recommended; the code automatically falls back to CPU.

### MINDTCT (optional, only for `extract_patches.py`)

`xfpad.data.patches` and `scripts/extract_patches.py` rely on the **NIST MINDTCT** binary. Place the executable (`mindtct` on Linux/macOS, `mindtct.exe` on Windows) in the repo root, or pass `--binary /path/to/mindtct`, or set `XFPAD_MINDTCT=/path/to/mindtct` in the environment. MINDTCT is part of NBIS and is freely redistributable; we do not bundle it here.

---

## Data preparation

The pipeline ingests two split files per scanner — one for the training PAIs, one for the unseen-PAI validation set — placed at:

```
data/splits/<scanner>_train.txt
data/splits/<scanner>_test.txt
```

Each file contains one image path per line. Paths are matched against the substring rules declared in the scanner config (see `configs/greenbit.yaml`/`configs/dermalog.yaml`) to assign integer labels: `0` for bona fide, `1..K` for the K known PAIs (training split), or `0` plus `1..U` for unseen PAIs (test split).

If your raw images are full fingerprints rather than 224×224 patches, run the extractor first:

```bash
python scripts/extract_patches.py \
    --split-file data/splits/greenbit_train.txt \
    --output-root images/greenbit_train
```

Then update the split files to point to the extracted patches.

---

## Reproducing the paper

The full pipeline is split into three numbered phases that mirror Section IV-A of the paper. All commands assume you are at the repo root with the virtual environment active. Replace `greenbit` with `dermalog` to reproduce the Dermalog results.

### Phase 1 — Manifold construction

Three sub-stages, runnable independently or all together via `--stage all`.

```bash
# 1.1 train MobileNet-v2 multi-class backbone f_phi
python scripts/phase1_train.py -c configs/greenbit.yaml --stage backbone

# 1.2 cache 1280-D embeddings for both train and test splits
python scripts/phase1_train.py -c configs/greenbit.yaml --stage features

# 1.3 train the geometric encoder g_psi, 10 seeds (paper: mean ± std over 10 runs)
python scripts/phase1_train.py -c configs/greenbit.yaml --stage encoder --num-runs 10
```

Or, for a clean room run:

```bash
python scripts/phase1_train.py -c configs/greenbit.yaml --stage all --num-runs 10
```

### (Optional) Circular ordering

Once features are cached, you can find the cosine-similarity-optimal cyclic label permutation:

```bash
python scripts/optimize_ordering.py -c configs/greenbit.yaml \
    --save-json outputs/ordering_greenbit.json
```

The script prints a label-reassignment report; transcribe the new assignment into your scanner config and re-run Phase 1.3 if you want the angularly-coherent layout shown in Figs. 4 and 5 of the paper.

### Phase 2 — Directional mapping of unseen PAIs

```bash
# Aggregate over the 10 trained encoders, produce Phase 2 plots and JSON.
python scripts/phase2_map_unseen.py -c configs/greenbit.yaml \
    --num-runs 10 --plot \
    --save-json outputs/phase2_greenbit.json
```

This produces:

- `outputs/projections/greenbit/seed*.npz` — projected `z_train` / `z_test`;
- `outputs/plots/greenbit/seed*/training.png` — Fig. 4(a) / 5(a);
- `outputs/plots/greenbit/seed*/unseen_<class>.png` — Fig. 4(b)–(g) / 5(b)–(e);
- a printed Phase 2 table (Table III in the paper).

For the cross-sensor preliminary experiment mentioned in Section VI, pass a different scanner config to `--test-config`:

```bash
python scripts/phase2_map_unseen.py -c configs/greenbit.yaml \
    --test-config configs/dermalog.yaml --num-runs 10
```

### Phase 3 — Targeted ablation (Table IV)

For each backbone × ablation pair, train 10 seeds and then evaluate them.

Baseline (no ablation):

```bash
python scripts/phase3_audit_pad.py -c configs/greenbit.yaml \
    --backbone mobilenet_v2 --action both --num-runs 10
```

A targeted ablation, e.g. removing all training paths whose name contains `Wood`:

```bash
python scripts/phase3_audit_pad.py -c configs/greenbit.yaml \
    --backbone mobilenet_v2 --action both --num-runs 10 \
    --ablate Wood --ablation-name without_wood_glue \
    --save-json outputs/phase3_greenbit_mobilenet_without_wood_glue.json
```

`--ablate` accepts multiple substrings, which act as OR (any path matching at least one substring is removed). The exact ablation set used in the paper is:

| Sensor   | Ablation             | `--ablate` argument |
|----------|----------------------|---------------------|
| Green Bit | Without Wood Glue    | `Wood`              |
| Green Bit | Without Latex        | `Latex`             |
| Green Bit | Without RPro10       | `RPRO10`            |
| Dermalog  | Without RProFast     | `Fast`              |
| Dermalog  | Without Latex        | `Latex`             |
| Dermalog  | Without RPro10       | `RPRO10`            |

Repeat with `--backbone resnet18` and `--backbone densenet121` to populate the full Table IV.

---

## Hyperparameter reference

| Parameter | Value | Where |
|---|---|---|
| Bona fide radius `rho_bf` | 1.0 | `configs/base.yaml: loss.rho_bf` |
| Radial margin `Delta rho` | 1.0 | `configs/base.yaml: loss.delta_rho` |
| Decision threshold `T = rho_bf^2` | 1.0 | derived |
| `gamma_min, gamma_max` | 1, 1000 | `configs/base.yaml: loss` |
| Inverse temperature `tau` | 5.0 | `configs/base.yaml: attribution.tau` |
| Backbone training | 200 epochs, Adam lr=1e-3, wd=1e-5, cosine schedule, batch 16 | `configs/base.yaml: backbone` |
| Encoder training | 200 epochs, Adam lr=1e-3, wd=1e-5, cosine schedule, batch 128, 10 seeds | `configs/base.yaml: geometric` |
| PAD detector training | 50 epochs, Adam lr=1e-3, wd=1e-5, cosine schedule, batch 16, 10 seeds | `configs/base.yaml: pad_detector` |
| Decision threshold (binary classifier) | 0.5 | `configs/base.yaml: pad_detector.threshold` |

The Δρ ablation (Appendix B of the paper) is performed by overriding `loss.delta_rho` in a custom config (e.g. `delta_rho: 0.1` or `10.0`) and re-running Phase 1.3 + Phase 2.

---

## Programmatic API

The package is fully usable from Python without going through the CLI. Example:

```python
import numpy as np, torch
from xfpad.config import load_config
from xfpad.models import FeatureExtractor, GeometricEncoder
from xfpad.metrics import calculate_metrics, analyze_unseen_pais
from xfpad.utils import resolve_device

cfg = load_config("configs/greenbit.yaml")
device = resolve_device(cfg.device)

fe = FeatureExtractor(in_channels=1, training_mode=False).to(device)
fe.load_state_dict(torch.load(f"checkpoints/feature_extractor_{cfg.scanner}.pth")["model"], strict=False)
fe.eval()

ge = GeometricEncoder().to(device)
ge.load_state_dict(torch.load(f"checkpoints/geometric_{cfg.scanner}_0.pth")["model"])
ge.eval()

# z_train, labels_train = ...  # produced via _project() in scripts/phase2_map_unseen.py
# bfo, rci, acs = calculate_metrics(z_train, labels_train, rho_bf=cfg.loss.rho_bf)
```

See `scripts/phase2_map_unseen.py:_project` for a worked example of running both stages over a list of paths.

---

## Paper artefact map

| Paper element | Source |
|---|---|
| Eq. (1) `L_conc` | `xfpad/losses/concentric.py` |
| Eq. (2)–(3) `L_cos` | `xfpad/losses/angular.py` |
| Eq. (4) `S(x)` log-scaling | `xfpad/losses/angular.py: make_log_scale` |
| Eq. (5) `p_{u,k}` attribution | `xfpad/metrics/attribution.py: analyze_unseen_pais` |
| Eq. (6)–(8) BFO / RCI / ACS | `xfpad/metrics/geometric.py: calculate_metrics` |
| Fig. 1 pipeline | `scripts/phase1_train.py` |
| Fig. 2 encoder architecture | `xfpad/models/geometric_encoder.py` |
| Fig. 4 / Fig. 5 manifolds | `scripts/phase2_map_unseen.py --plot` |
| Fig. 6 Δρ ablation | re-run phase 1.3 with overridden `loss.delta_rho` |
| Table I / II datasets | `data/splits/*.txt` |
| Table III Phase 2 anchors | `phase2_map_unseen.py --num-runs 10 --save-json` |
| Table IV Phase 3 ablation | `phase3_audit_pad.py --action both --num-runs 10` |
| Table V Δρ metrics | `xfpad/metrics/geometric.py` invoked via Phase 2 |

---

## Citation

```bibtex
@article{carta2025xfpad,
  title   = {The Fingerprint {PAD} Exposimeter ({X-FPAD}):
             A Visual Framework for Evaluating Generalization to Unseen Attacks},
  author  = {Carta, Simone and Casula, Roberto and Marcialis, Gian Luca},
  year    = {2025},
}
```

---

## License

Released under the MIT License (see `LICENSE`).

---

## Acknowledgements

This work makes use of the LivDet 2019, LivDet 2021 (including the ScreenSpoof variants), and LivDet 2023 datasets. We are grateful to the LivDet organisers and to the broader fingerprint biometrics community.
