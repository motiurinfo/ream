"""
recompute_without_K3.py  —  KronoDroid reliability WITHOUT the degenerate K3_Static.

Why: K3_Static has only ~9-13 features, so a top-10 Jaccard spans the whole
feature set and every agreement/stability value collapses to 1.000. Those cells
are not real results. This script recomputes KronoDroid's cross-modal reliability
from K1_Syscall + K2_Perm only. K3_Static stays valid for detection-F1 (reported
separately), just not for the Jaccard-based reliability metrics.

With only 2 modalities, the Friedman test (used for the 3-modality datasets) does
not apply — it needs >=3 groups. The correct paired two-group test is the
Wilcoxon signed-rank test, with a matched-pairs rank-biserial effect size.

Usage (from the project root, same place you run the notebooks):
    python recompute_without_K3.py
or inside a notebook cell:
    import sys; sys.path.insert(0, '..')
    from recompute_without_K3 import recompute
    recompute(results_dir='../results')

Outputs, written next to the originals in results/<dataset>/:
    <dataset>_K1vsK2_wilcoxon.csv   # per model x metric: means, gap, p, effect size
and a combined results/kronodroid_modality_effect_noK3.csv for the paper table.
"""
import os, json, glob
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

KEEP = ["K1_Syscall", "K2_Perm"]          # K3_Static dropped (degenerate)
DATASETS = ["krono_real", "krono_emu"]
METRICS = ["stab_adjjac", "stab_rbo", "stab_rho",
           "del_median", "shap_lime_adjjac_disc1", "intr_shap_adjjac"]


def _stars(p):
    return "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else "ns"))


def _load_cells(results_dir, dataset, modality):
    """Load the cells list from results/<dataset>/<dataset>_<modality>_fixed.json."""
    path = os.path.join(results_dir, dataset, f"{dataset}_{modality}_fixed.json")
    if not os.path.exists(path):                       # fallback: search recursively
        hits = glob.glob(os.path.join(results_dir, "**",
                                      f"{dataset}_{modality}_fixed.json"), recursive=True)
        if not hits:
            raise FileNotFoundError(f"missing {dataset}_{modality}_fixed.json under {results_dir}")
        path = hits[0]
    return json.load(open(path))["cells"]


def _paired_vectors(cbm, model, metric):
    """Return matched (K1, K2) value arrays, paired by (seed, instance idx)."""
    t1, t2 = KEEP
    table = {}
    for t in (t1, t2):
        for c in cbm[t]:
            if c["model"] != model:
                continue
            for r in c["records"]:
                if metric in r:
                    table.setdefault((c["seed"], r["idx"]), {})[t] = r[metric]
    a = np.array([v[t1] for v in table.values() if t1 in v and t2 in v])
    b = np.array([v[t2] for v in table.values() if t1 in v and t2 in v])
    return a, b


def recompute(results_dir="results", verbose=True):
    combined = []
    for ds in DATASETS:
        cbm = {m: _load_cells(results_dir, ds, m) for m in KEEP}
        models = sorted({c["model"] for c in cbm[KEEP[0]]})
        rows = []
        for model in models:
            for metric in METRICS:
                a, b = _paired_vectors(cbm, model, metric)
                if len(a) < 10:
                    continue
                diff = a - b
                if np.allclose(diff, 0):
                    W, p, r_rb = 0.0, 1.0, 0.0
                else:
                    res = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
                    W, p = float(res.statistic), float(res.pvalue)
                    n = int(np.sum(diff != 0))
                    r_rb = 1 - (2 * W) / (n * (n + 1) / 2)   # rank-biserial effect size
                rows.append(dict(dataset=ds, model=model, metric=metric,
                                 n_pairs=int(len(a)),
                                 mean_K1=round(float(a.mean()), 4),
                                 mean_K2=round(float(b.mean()), 4),
                                 mean_gap=round(float(a.mean() - b.mean()), 4),
                                 wilcoxon_p=p, rank_biserial=round(float(r_rb), 4),
                                 sig=_stars(p)))
        df = pd.DataFrame(rows)
        out = os.path.join(results_dir, ds, f"{ds}_K1vsK2_wilcoxon.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df.to_csv(out, index=False)
        if verbose:
            print(f"\n=== {ds}: K1 vs K2 stability (K3 excluded) ===")
            print(df[df.metric == "stab_adjjac"][
                ["model", "n_pairs", "mean_K1", "mean_K2", "mean_gap", "rank_biserial", "sig"]
            ].to_string(index=False))
        # headline modality effect = mean over tree models of stab_adjjac |rank-biserial|
        tree = df[(df.metric == "stab_adjjac") & (df.model.isin(["lgbm", "rf"]))]
        combined.append(dict(dataset=ds,
                             stab_gap_tree=round(tree.mean_gap.abs().mean(), 4),
                             stab_effect_tree=round(tree.rank_biserial.abs().mean(), 4),
                             test="wilcoxon(2-modality)"))
    comb = pd.DataFrame(combined)
    comb_path = os.path.join(results_dir, "kronodroid_modality_effect_noK3.csv")
    comb.to_csv(comb_path, index=False)
    if verbose:
        print("\n=== KronoDroid modality effect, K3-excluded (for paper table) ===")
        print(comb.to_string(index=False))
        print(f"\nsaved per-dataset *_K1vsK2_wilcoxon.csv and {comb_path}")
    return comb


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", help="path to results/ dir")
    recompute(ap.parse_args().results)
