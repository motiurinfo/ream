"""Contribution 4 — temporal / concept-drift reliability.

Does explanation reliability degrade when the detector faces NEWER malware than it
was trained on? Needs a timestamped dataset (AndroZoo / KronoDroid); CCCS and
MalDroid lack usable timestamps, so this module is dataset-agnostic and expects a
DataFrame with a `year` (or timestamp) column.

Design: temporal split — train on apps up to year T, evaluate faithfulness &
stability on apps from year > T. Compare against a random (non-temporal) split of
equal sizes. If reliability drops under the temporal split but not the random one,
the drop is attributable to drift, not sample size.
"""
import numpy as np, pandas as pd
from ..core.models import train_model
from ..core.explainers import ShapExplainer
from ..core.metrics import (detect_feature_type, perturbation_stability,
    make_baselines, resolve_baseline, deletion_insertion_auc)


def temporal_split(df, year_col, cutoff):
    """Return boolean masks (train_old, test_new) by year cutoff."""
    yr=pd.to_numeric(df[year_col], errors="coerce")
    return (yr<=cutoff).to_numpy(), (yr>cutoff).to_numpy()


def evaluate_reliability(clf, model_kind, X_train_np, X_eval, y_eval, is_binary, d,
                         n=100, seed=0):
    """Mean SHAP faithfulness (median baseline) + stability over n eval instances."""
    se=ShapExplainer(clf, model_kind, X_background=X_train_np)
    rng=np.random.default_rng(seed)
    Xe=np.asarray(X_eval); ye=np.asarray(y_eval)
    idx=rng.choice(len(Xe), size=min(n,len(Xe)), replace=False)
    base=np.median(X_train_np,axis=0)
    dels, stabs = [], []
    for i in idx:
        x=Xe[i]; c=int(ye[i]); attr=se.attributions(x,c)
        dd,_=deletion_insertion_auc(clf,x,attr,c,base)
        aj,_,_=perturbation_stability(lambda z: se.attributions(z,c),x,is_binary,d,seed=int(i))
        dels.append(dd); stabs.append(aj)
    return dict(mean_deletion_auc=float(np.mean(dels)), mean_stability=float(np.mean(stabs)), n=len(idx))


def run_temporal(df, feature_cols, label_col, year_col, cutoff, model_kind="lgbm", seed=0):
    """Full temporal-vs-random comparison for one modality.
    df: rows with features + label + year. Returns dict with both conditions."""
    from sklearn.model_selection import train_test_split
    X=df[feature_cols].to_numpy(float); y=df[label_col].to_numpy()
    is_binary=detect_feature_type(X); d=X.shape[1]

    # temporal
    m_old,m_new=temporal_split(df,year_col,cutoff)
    clf_t=train_model(model_kind, X[m_old], y[m_old], seed=seed)
    temporal=evaluate_reliability(clf_t,model_kind,X[m_old],X[m_new],y[m_new],is_binary,d,seed=seed)

    # random split of equal train size
    n_old=int(m_old.sum())
    Xtr,Xte,ytr,yte=train_test_split(X,y,train_size=n_old,random_state=seed,stratify=y)
    clf_r=train_model(model_kind,Xtr,ytr,seed=seed)
    random_=evaluate_reliability(clf_r,model_kind,Xtr,Xte,yte,is_binary,d,seed=seed)

    return dict(cutoff=cutoff, model=model_kind,
                temporal=temporal, random=random_,
                stability_drop=random_["mean_stability"]-temporal["mean_stability"],
                faithfulness_change=temporal["mean_deletion_auc"]-random_["mean_deletion_auc"])
