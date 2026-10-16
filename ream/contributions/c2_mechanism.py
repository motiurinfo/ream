"""Contribution 2 — mechanism: WHEN does modality-dependence appear?

Correlate per-(dataset,modality) and per-pair properties with the observed
modality-effect size (Kendall's W). Converts non-replication into a condition.
"""
import os
import numpy as np, pandas as pd
from scipy.stats import entropy as shannon_entropy
from ..core import data as D
from ..core import config as C


def feature_sparsity(X): X=np.asarray(X); return float((X==0).mean())

def mean_feature_entropy(X, bins=10):
    X=np.asarray(X); ents=[]
    for j in range(X.shape[1]):
        col=X[:,j]
        if np.all(col==col[0]): ents.append(0.0); continue
        h,_=np.histogram(col,bins=bins); p=h[h>0]/h.sum(); ents.append(float(shannon_entropy(p)))
    return float(np.mean(ents))

def modality_properties(X):
    X=np.asarray(X,dtype=float)
    return dict(dimensionality=int(X.shape[1]), sparsity=feature_sparsity(X),
                mean_entropy=mean_feature_entropy(X))

def pair_redundancy(Xa, Xb, top=20):
    Xa=np.asarray(Xa,float); Xb=np.asarray(Xb,float)
    va=np.argsort(-Xa.var(0))[:top]; vb=np.argsort(-Xb.var(0))[:top]
    A=Xa[:,va]; B=Xb[:,vb]
    A=(A-A.mean(0))/(A.std(0)+1e-9); B=(B-B.mean(0))/(B.std(0)+1e-9)
    return float((np.abs(A.T@B)/len(A)).mean())


def collect_dataset(dataset, results_dir=None):
    """Compute modality drivers from feature CSVs + effect sizes from paired CSV."""
    results_dir=results_dir or os.path.join(C.RESULTS_DIR,dataset)
    tags=D.modalities_for(dataset)
    Xs={t:D.load_modality(dataset,t)[0].to_numpy() for t in tags}
    props={t:modality_properties(Xs[t]) for t in tags}
    pairs={}
    for i,a in enumerate(tags):
        for b in tags[i+1:]:
            pairs[f"{a}~{b}"]=dict(redundancy=pair_redundancy(Xs[a],Xs[b]),
                dim_ratio=max(props[a]["dimensionality"],props[b]["dimensionality"])/
                          max(1,min(props[a]["dimensionality"],props[b]["dimensionality"])))
    effect={}
    # KronoDroid: K3_Static is degenerate (d<2k, every top-k overlap is 1.0), so the
    # 3-modality Kendall's W in *_paired_significance.csv is contaminated. Prefer the
    # K3-clean 2-modality (K1 vs K2) Wilcoxon rank-biserial, which is what the paper uses.
    clean=os.path.join(results_dir,f"{dataset}_K1vsK2_wilcoxon.csv")
    if "krono" in dataset and os.path.exists(clean):
        dfw=pd.read_csv(clean)
        col="rank_biserial" if "rank_biserial" in dfw.columns else dfw.columns[-1]
        for _,r in dfw.iterrows():
            metric=r.get("metric","stab_adjjac")
            effect[f"{r['model']}|{metric}"]=abs(float(r[col]))
    else:
        f=os.path.join(results_dir,f"{dataset}_paired_significance.csv")
        if os.path.exists(f):
            df=pd.read_csv(f)
            for _,r in df.iterrows(): effect[f"{r['model']}|{r['metric']}"]=float(r["kendall_w"])
    return dict(dataset=dataset,modality_props=props,pair_props=pairs,effect_sizes=effect)


def driver_table(collected):
    rows=[]
    for c in collected:
        props=c["modality_props"]; pairs=c["pair_props"]; eff=c["effect_sizes"]
        # tree models only (lgbm, rf), stability metric, to match the paper's headline effect
        w=[v for k,v in eff.items()
           if k.endswith("stab_adjjac") and k.split("|")[0] in ("lgbm","rf")]
        rows.append(dict(dataset=c["dataset"],
            mean_sparsity=round(np.mean([p["sparsity"] for p in props.values()]),3),
            mean_entropy=round(np.mean([p["mean_entropy"] for p in props.values()]),3),
            dim_spread=round(max(p["dimensionality"] for p in props.values())/
                             max(1,min(p["dimensionality"] for p in props.values())),2),
            mean_redundancy=round(np.mean([pp["redundancy"] for pp in pairs.values()]),3) if pairs else np.nan,
            modality_effect_W=round(np.mean(w),3) if w else np.nan))
    return pd.DataFrame(rows)
