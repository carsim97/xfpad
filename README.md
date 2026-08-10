# X-FPAD: Fingerprint PAD Exposimeter

Reference implementation for the paper

> **The Fingerprint PAD Exposimeter (X-FPAD): A Visual Framework for Evaluating Generalization to Unseen Attacks.**
> Simone Carta, Roberto Casula, Gian Luca Marcialis. University of Cagliari.

X-FPAD reformulates fingerprint Presentation Attack Detection (PAD) as a structured-manifold learning problem. A frozen MobileNet-v2 backbone produces 1280-D fingerprint embeddings; a lightweight MLP (the geometric encoder) projects them into a deterministic 2-D latent space governed by a radial-angular geometry — bona fide samples are anchored at the origin, each known PAI occupies a dedicated angular sector, and radial distance encodes deviation from authentic skin. The induced manifold is used as a diagnostic surface: unseen PAIs are projected onto it and their dominant angular anchors are identified by direct inspection or via the soft-attribution metric `p_{u,k}`.

The manifold is validated by removing the materials it names and retraining four PAD systems — MobileNet-v2, ResNet-18, DenseNet-121 and CFD-PAD — over an exhaustive leave-one-material-out sweep, and by running the same attribution forward to choose which absent material to acquire next.

---

## Installation

Tested with Python 3.9.

```bash
git clone https://github.com/carsim97/xfpad.git
cd xfpad
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .
```

The package installs `torch`, `torchvision`, `numpy`, `opencv-python`, `Pillow`, `matplotlib`, `tqdm`, `PyYAML`, `scipy` and `scikit-learn` (versions pinned in `requirements.txt`).

A CUDA-enabled GPU is recommended; the code falls back to CPU automatically.

Each scanner has two configurations: `configs/<scanner>.yaml` and a `_cons` variant identical to it except that every DataLoader runs at `num_workers: 0`. The scanner name is the same in both, so checkpoints and outputs land in the same place and results are bit-identical; only the data-loading path differs. The runs reported in the paper used the `_cons` variants.

### MINDTCT (only for `extract_patches.py`)

`xfpad.data.patches` and `scripts/extract_patches.py` rely on the **NIST MINDTCT** binary. Place the executable (`mindtct` on Linux/macOS, `mindtct.exe` on Windows) in the repo root, pass `--binary /path/to/mindtct`, or set `XFPAD_MINDTCT` in the environment. MINDTCT is part of NBIS and is freely redistributable; it is not bundled here.

---

## Data availability

The split files are **not** included. They list image paths only, but they index the **LivDet 2019 / 2021 / 2023** datasets, distributed by the LivDet organizers under their own access terms. To reproduce the experiments, obtain the LivDet data from the official sources and create one plain-text file per scanner and split:

```
data/splits/<scanner>_train.txt   # known (training) PAIs
data/splits/<scanner>_test.txt    # unseen (validation) PAIs
```

Each line is the path to one image. Paths are mapped to integer labels via the substring rules in `configs/<scanner>.yaml` (`0` = bona fide, `1..K` = known PAIs in the train split; `0` plus `1..U` = unseen PAIs in the test split).

If the raw images are full fingerprints rather than 224×224 patches, extract them first and repoint the split files:

```bash
python scripts/extract_patches.py \
    --split-file data/splits/greenbit_train.txt \
    --output-root images/greenbit_train
```

---

## Reproducing the paper

Four phases, mirroring Section IV-A. Commands assume the repo root and an active environment; replace `greenbit` with `dermalog` for the second sensor.

### Phase 1 — Manifold construction

```bash
# backbone f_phi, cached 1280-D embeddings, and the geometric encoder over 10 seeds
python scripts/phase1_train.py -c configs/greenbit.yaml --stage all --num-runs 10
```

The three stages (`--stage backbone|features|encoder`) can also be run separately.

### Phase 2 — Directional mapping of unseen PAIs

```bash
python scripts/phase2_map_unseen.py -c configs/greenbit.yaml \
    --num-runs 10 --plot --save-json outputs/phase2_greenbit_intra.json
```

This writes the per-seed projections, the latent-space panels and the attribution weights `p_{u,k}` that the rest of the protocol consumes. Passing `--test-config configs/dermalog.yaml` projects one sensor's unseen PAIs onto the other's manifold, which is the cross-sensor experiment discussed as a limitation in Section VI.

### Phase 3a — Exhaustive leave-one-material-out sweep

Every known material of both sensors is removed in turn, the four audited systems are retrained from scratch over 10 seeds, and APCER is measured per unseen PAI. One driver enumerates the whole matrix and dispatches each cell to the right trainer:

```bash
# baseline + sweep + matched-size controls, one worker per GPU
python scripts/lomo_driver.py --blocks baseline lomo controls --num-runs 10 --gpus 0 1

# print the queue without launching anything
python scripts/lomo_driver.py --dry-run
```

The driver is idempotent: a cell whose output JSON exists is skipped, so it resumes after an interruption. Individual cells can also be run directly with `scripts/phase3_audit_pad.py --backbone <name>` or, for CFD-PAD, with `scripts/phase3_cfdpad.py`.

Once the cells are on disk, the analysis pairs each attribution weight with the shift it predicted:

```bash
python scripts/correlate_attr_apcer.py --save-json outputs/point_b_correlation.json
python scripts/ablation_controls.py   --save-json outputs/point_c_controls.json
```

### Phase 3b — Manifold-guided dataset composition

Part of each training vocabulary is withheld; the encoder is retrained on what remains and asked which withheld material to reinstate.

```bash
# --keep lists the retained vocabulary: Green Bit withholds the three materials
# represented by 400 samples, Dermalog withholds the Latex family.
python scripts/train_reduced_encoder.py -c configs/greenbit.yaml --num-runs 10 \
       --keep Latex "Latex V2" RProFast RPro10
python scripts/lomo_driver.py --blocks phase4 --num-runs 10 --gpus 0 1

python scripts/phase4_analysis.py      --save-json outputs/phase4_analysis.json
python scripts/phase4_shared_anchor.py --save-json outputs/point_e_shared_anchor.json
python scripts/phase4_ranking.py
```

### Diagnostic baselines

The same reading is applied to other representations of the same frozen embedding — the 1280-D space itself, PCA, t-SNE, UMAP, an encoder trained without the angular term, and free prototypes under CosFace and ArcFace margins:

```bash
python scripts/train_encoder_radial_only.py -c configs/greenbit.yaml --num-runs 10
python scripts/train_encoder_arcface.py     -c configs/greenbit.yaml --num-runs 10 --variant cosface
python scripts/train_encoder_arcface.py     -c configs/greenbit.yaml --num-runs 10 --variant arcface

python scripts/baselines_attribution.py    -c configs/greenbit.yaml \
       --save-json outputs/baselines_greenbit.json
python scripts/baselines_predictiveness.py --save-json outputs/point_a_predictiveness.json
python scripts/cosface_layout_stability.py
```

### Stability of the reading

Materials are bound to angular sectors by label index, so the attribution is re-derived under nine alternative assignments: eight random permutations and the cyclic ordering a similarity optimiser would choose, which is the one most likely to expose an artefact of the arbitrary order.

```bash
# the similarity-optimal ordering, used as one of the challenger layouts
python scripts/optimize_ordering.py -c configs/greenbit.yaml \
    --save-json outputs/ordering_greenbit.json

python scripts/permutation_stability.py -c configs/greenbit.yaml   # prototype permutation
python scripts/phase2_run_stability.py                             # agreement across the 10 encoders
```

---

## Verifying the published numbers

Every table in the paper is emitted from the analysis JSONs rather than typed by hand, and every emitter carries a `--selftest` that asserts the values as published. Running the selftest is the fastest way to check that a re-run reproduces the paper:

```bash
python scripts/emit_phase4_table.py --selftest
# selftest ok — start 52.2, X-FPAD 33.8, oracle 31.1 (87% of the maximum)
```

| Paper element | Produced by | Verified by |
|---|---|---|
| Eq. (1) `L_conc` | `xfpad/losses/concentric.py` | — |
| Eq. (2) `φ_k`, Eq. (3) `L_cos` | `xfpad/losses/angular.py` | — |
| Eq. (S1) `p_{u,k}` | `xfpad/metrics/attribution.py: analyze_unseen_pais` | — |
| Eq. (S2)–(S4) BFO / RCI / ACS | `xfpad/metrics/geometric.py: calculate_metrics` | — |
| Eq. (S5) `S(x)` log-scaling | `xfpad/losses/angular.py: make_log_scale` | — |
| Fig. 1 pipeline, Fig. 2 contrast | conceptual illustrations (no script) | — |
| Fig. 3 Phase 2 projections | `scripts/replot_phase2.py` | `--selftest` |
| Fig. 4 sweep scatter | `scripts/plot_phase3_scatter.py` | `--selftest` |
| Fig. S1 Δρ latent spaces | Phase 1 with `loss.delta_rho` overridden | — |
| Figs. S2–S3 remaining projections | `scripts/replot_phase2.py` | `--selftest` |
| Tables I–II datasets | `data/splits/*.txt` | — |
| Table III Phase 2 anchors | `scripts/emit_phase2_tables.py` | `--selftest` |
| Table IV Phase 3a ablations | `scripts/emit_phase3_table.py` | `--selftest` |
| the 920 detector trainings the ablation tables read | `scripts/phase3_audit_pad.py`, `scripts/phase3_cfdpad.py`, dispatched by `scripts/lomo_driver.py` | — |
| Table V Phase 3b strategies | `scripts/emit_phase4_table.py` | `--selftest` |
| Table S1 Δρ metrics | `xfpad/metrics/geometric.py` via Phase 2 | — |
| Table S2 Phase 2, full | `scripts/emit_phase2_tables.py` | `--selftest` |
| Table S3 sweep and baselines | `scripts/emit_supp_sweep.py`, `scripts/emit_supp_baselines.py` | `--selftest` |
| Table S4 noise-floor sensitivity | `scripts/emit_supp_thresholds.py` | `--selftest` |
| Table S5 prototype permutation | `scripts/emit_supp_permutation.py` | `--selftest` |
| the permutations it reads | `scripts/permutation_stability.py`, `scripts/optimize_ordering.py` | — |
| Table S6 Phase 3b per unseen PAI | `scripts/emit_supp_phase3b.py` | `--selftest` |
| Table S7 alternative representations | `scripts/emit_supp_representations.py` | `--selftest` |
| Table S8 cross-sensor attribution | `scripts/emit_supp_crosssensor.py` | `--selftest` |
| ρ over the 58 analysis units (Sec. V-C) | `scripts/correlate_attr_apcer.py` | — |
| ρ over the 26 candidate–PAI pairs (Sec. V-D) | `scripts/phase4_ranking.py` | `--selftest` |
| matched-size random controls (Sec. V-C) | `scripts/ablation_controls.py` | `--selftest` |
| CosFace layout reproducibility (Sec. S7) | `scripts/cosface_layout_stability.py` | `--selftest` |

---

## Audited systems

Four PAD systems are retrained across every ablation of the protocol, and one analysis unit is the average of their four readings.

`scripts/phase3_audit_pad.py` trains the three generic architectures, selected with `--backbone mobilenet_v2|resnet18|densenet121`. CFD-PAD has its own entry point, `scripts/phase3_cfdpad.py`, because its three-term objective and its channel-importance pass do not fit that dispatch; it is reimplemented from its published description, with the importance pass at every batch and 30 of 160 denoised channels. Splits, labels, dataset caching, APCER evaluation and JSON layout are shared between the two, so the cells are directly comparable, and `scripts/lomo_driver.py` dispatches to whichever is needed.

The list of audited systems and the two thresholds the reading rests on (`p_{u,k} ≥ 0.30`, `|Δ/σ| < 1.2`) are defined once, in `scripts/_protocol.py`.

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

The Δρ ablation of Section S2 is performed by overriding `loss.delta_rho` in a custom config (e.g. `delta_rho: 0.1` or `10.0`) and re-running Phase 1 and Phase 2.

---

## Programmatic API

The package is usable from Python without the CLI:

```python
import torch
from xfpad.config import load_config
from xfpad.models import FeatureExtractor, GeometricEncoder
from xfpad.metrics import calculate_metrics, analyze_unseen_pais
from xfpad.utils import resolve_device

cfg = load_config("configs/greenbit.yaml")
device = resolve_device(cfg.device)

fe = FeatureExtractor(in_channels=1, training_mode=False).to(device)
fe.load_state_dict(torch.load(f"checkpoints/feature_extractor_{cfg.scanner}.pth")["model"],
                   strict=False)
fe.eval()

ge = GeometricEncoder().to(device)
ge.load_state_dict(torch.load(f"checkpoints/geometric_{cfg.scanner}_0.pth")["model"])
ge.eval()

# z_train, labels_train = ...  # see _project() in scripts/phase2_map_unseen.py
# bfo, rci, acs = calculate_metrics(z_train, labels_train, rho_bf=cfg.loss.rho_bf)
```

---

## Citation

```bibtex
@article{carta2026xfpad,
  author  = {Carta, Simone and Casula, Roberto and Marcialis, Gian Luca},
  title   = {The Fingerprint {PAD} Exposimeter ({X-FPAD}): A Visual Framework
             for Evaluating Generalization to Unseen Attacks},
  journal = {IEEE Transactions on Biometrics, Behavior, and Identity Science},
  year    = {2026},
  note    = {Under review},
}
```

---

## License

Released under the MIT License (see `LICENSE`).

---

## Acknowledgements

This work makes use of the LivDet 2019, LivDet 2021 (including the ScreenSpoof variants), and LivDet 2023 datasets. We are grateful to the LivDet organisers and to the broader fingerprint biometrics community.
