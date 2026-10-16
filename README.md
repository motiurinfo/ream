# REAM — Reliability-Audit Framework for Explainable Malware Detection

`ream` is the reference implementation of **REAM**, a deployment-time framework that
audits whether the post-hoc explanations (SHAP, LIME) of an Android malware detector are
reliable enough to be trusted on the specific data and model in use. Given a trained
detector, its feature modality, and a labelled sample, REAM returns a per-deployment
**reliability verdict** — a pass/fail decision with chance-corrected stability,
multi-baseline faithfulness, and explainer-agreement estimates, each with a seed-level
confidence interval.

This repository accompanies the paper:

> S. S. M. M. Rahman and T. Islam, *"REAM: A Reliability-Audit Framework for Trustworthy
> Explanations in Android Malware Detection,"* submitted to **Expert Systems with
> Applications** (Elsevier), 2026.

## What REAM does

REAM unifies four corrected measurement components into a single six-step audit:

| Stage | Component | Fixes |
|-------|-----------|-------|
| 1. Sampling gate | Common-correct sample selection (P4) | Every configuration judged on the same inputs |
| 2. Faithfulness probe | Multi-baseline deletion/insertion AUC | Single-baseline faithfulness failure |
| 3. Stability probe | Modality-aware perturbation (P1) + chance-corrected, size-robust agreement (P2) | Non-comparable perturbation; dimensionality-driven chance overlap |
| 4. Decision | Paired tests with effect sizes (P3); resolve SHAP–LIME disagreement | Unpaired testing; LIME-discretization disagreement |

## Structure
```
ream/
  core/
    config.py       # all paths & experiment scale — EDIT HERE
    data.py         # MalDroid + CCCS (dynamic) loaders
    kronodroid.py   # KronoDroid (real/emu) loader
    models.py       # LightGBM / RF / MLP + intrinsic logistic
    explainers.py   # SHAP (tree/kernel) + LIME (both discretizations)
    metrics.py      # corrected metrics (P1–P4) + multi-baseline faithfulness
  contributions/
    c1_reliability.py   # the REAM audit pipeline (two-pass grid)
    c2_mechanism.py     # when does modality-dependence appear (drivers vs effect)
    c3_adversarial.py   # prediction-preserving explanation-destabilization attack
    c4_temporal.py      # concept-drift reliability (needs timestamped data)
    w6_numbers.py       # single-source-of-truth number/macro generator
  run.py            # CLI entry point
notebooks/          # thin drivers: 00 setup ... 06 kronodroid
data/               # put dataset files here (see below)
results/            # outputs land here
```

## Datasets
The three source datasets are third-party and **not** redistributed here; obtain them
from their original providers and place them under `data/`:
- **CICMalDroid-2020**: the 3 feature CSVs + `syscall_unique` in `data/`.
- **CCCS-CIC-AndMal-2020 (dynamic)**: the `*_after_reboot_Cat.csv` files anywhere under
  `data/` (loader searches recursively). Static files are not usable cross-modally.
- **KronoDroid (real + emu parquet)**: `kronodroid-2021-real-v1.parquet` and/or
  `-emu-v1.parquet` under `data/`. Enables 3 modalities (K1_Syscall / K2_Perm /
  K3_Static), concept-drift (timestamps), and cross-device (emu vs real) — datasets
  `krono_real` / `krono_emu`.

Set `XAI_DATA_DIR` / `XAI_RESULTS_DIR` environment variables, or edit `core/config.py`.

## Quick start
```bash
pip install -r requirements.txt
python -m ream.run --dataset maldroid --quick     # validate end-to-end
python -m ream.run --dataset maldroid             # full audit (hours)
python -m ream.run --dataset cccs
python -m ream.run --dataset krono_real           # binary; 3 modalities + timestamps
python -m ream.run --dataset krono_emu
```
Or use the notebooks in order (00 → 06).

## Contributions → code map
1. The REAM audit (corrected components P1–P4) → `contributions/c1_reliability` and
   `core/metrics`.
2. Why the audit is necessary (drivers do not predict the effect) → `c2_mechanism`.
3. Adversarial explanation robustness → `c3_adversarial`.
4. Temporal / concept-drift reliability → `c4_temporal`.
5. Number/macro consistency automation → `w6_numbers`.

## Reproducibility
Fixed seeds; one stratified split per seed reused across modalities; deterministic
TreeSHAP; documented feature selection; seed-level confidence intervals; paired Friedman
+ Kendall's *W* (3 modalities) or Wilcoxon + rank-biserial (2 modalities) for
significance. Environment: Python 3.13, SHAP 0.50, LightGBM 4.6, scikit-learn 1.7,
NumPy 2.2, SciPy 1.16.

## Scale / runtime
Full grid = datasets × 3 models × 5 seeds × modalities. Budget hours on a workstation;
use `--quick` (or `config.quick_mode()`) to validate first. Random-forest TreeSHAP
dominates the cost (about 92% of total), so a gradient-boosted detector is far cheaper to
audit.

## Results consistency
All numbers in the paper are the **tree-model (LightGBM + RF)** effects on the
`stab_adjjac` stability metric. For KronoDroid, the reported modality effect is the
**K3-clean** rank-biserial (K1 vs K2), because the K3_Static block has fewer features than
the top-*k* size and its top-*k* overlaps are degenerate (always 1.0); see
`*_K1vsK2_wilcoxon.csv`. `results/mechanism_drivers.csv` uses these same tree-model,
K3-clean effects, so it matches the "Candidate drivers vs. observed modality effect" table
in the paper.

## License
Code is released under the MIT License (see `LICENSE`). The datasets
(CICMalDroid-2020, CCCS-CIC-AndMal-2020, KronoDroid) are third-party; obtain them from
their original providers under their own terms. No dataset files are included here.

## Citation
If you use this material, please cite the paper (BibTeX added on publication), this
repository / its Zenodo archive, and the three source datasets from their original
providers.
