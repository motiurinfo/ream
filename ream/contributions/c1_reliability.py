"""Contribution 1 — corrected reliability measurement.

Two-pass grid (W4): pass A finds common-correct apps across all (modality,model);
pass B explains only those, computing W1 modality-aware stability, W2
chance-corrected agreement, multi-baseline faithfulness, LIME both discretizations.
Saves one JSON per modality + aggregate/significance CSVs.
"""
import os, json, time
import numpy as np, pandas as pd
from scipy.stats import t as tdist
from ..core import config as C
from ..core import data as D
from ..core.models import train_model, train_intrinsic, intrinsic_attributions, evaluate
from ..core.explainers import ShapExplainer, LimeExplainer
from ..core.metrics import (detect_feature_type, perturbation_stability, adjusted_jaccard,
    make_baselines, resolve_baseline, deletion_insertion_auc, friedman_paired, sig_stars,
    common_correct_indices)


def run_cell(tag, model_kind, seed, Xtr, Xte, ytr, yte, cn, common_idx, is_binary,
             n_shap=C.N_SHAP, n_lime=C.N_LIME, baselines=C.BASELINES, lime_modes=C.LIME_MODES):
    Xtr_np,Xte_np=Xtr.to_numpy(),Xte.to_numpy(); ytr_np=ytr.to_numpy()
    rng=np.random.default_rng(seed); base_store=make_baselines(Xtr_np,ytr_np,baselines,rng); d=Xtr.shape[1]
    clf =train_model(model_kind,Xtr,ytr,seed=seed)
    clf2=train_model(model_kind,Xtr,ytr,seed=seed+100)
    intr=train_intrinsic(Xtr,ytr,seed=seed)
    ev=evaluate(clf,Xte,yte,cn,f"{tag}-{model_kind}")
    sel=common_idx
    if model_kind=="mlp" and len(sel)>max(20,n_shap//3): sel=rng.choice(sel,size=max(20,n_shap//3),replace=False)
    elif len(sel)>n_shap: sel=rng.choice(sel,size=n_shap,replace=False)
    se =ShapExplainer(clf ,model_kind,X_background=Xtr_np)
    se2=ShapExplainer(clf2,model_kind,X_background=Xtr_np)
    limes={dm:LimeExplainer(clf,Xtr_np,Xtr.columns,cn,discretize=dm,seed=seed) for dm in lime_modes}
    recs=[]
    for j,i in enumerate(sel):
        x=Xte_np[i]; c=int(yte.to_numpy()[i]); sa=se.attributions(x,c)
        rec=dict(idx=int(i),cls=c,cls_name=cn[c])
        for bn in baselines:
            bl=resolve_baseline(bn,base_store,Xtr_np,ytr_np,c,rng)
            dd,ins=deletion_insertion_auc(clf,x,sa,c,bl); rec[f"del_{bn}"]=dd; rec[f"ins_{bn}"]=ins
        aj,rb,sr=perturbation_stability(lambda z: se.attributions(z,c),x,is_binary,d,seed=int(i))
        rec["stab_adjjac"]=aj; rec["stab_rbo"]=rb; rec["stab_rho"]=sr
        rec["seed_adjjac"]=adjusted_jaccard(sa,se2.attributions(x,c),d=d)
        rec["intr_shap_adjjac"]=adjusted_jaccard(sa,intrinsic_attributions(intr,x,c),d=d)
        if j<n_lime:
            for dm in lime_modes:
                rec[f"shap_lime_adjjac_disc{int(dm)}"]=adjusted_jaccard(sa,limes[dm].attributions(x,c),d=d)
        recs.append(rec)
    return dict(tag=tag,model=model_kind,seed=seed,shape=list(Xtr.shape),
                macro_f1=ev["macro_f1"],weighted_f1=ev["weighted_f1"],confusion=ev["confusion"],
                intrinsic_macro_f1=evaluate(intr,Xte,yte,cn)["macro_f1"],
                n_explained=len(recs),n_common=int(len(common_idx)),records=recs)


def run_dataset(dataset, models=C.MODELS, seeds=C.SEEDS, out_dir=None,
                n_shap=C.N_SHAP, n_lime=C.N_LIME):
    out_dir=out_dir or os.path.join(C.RESULTS_DIR, dataset); os.makedirs(out_dir,exist_ok=True)
    tags=D.modalities_for(dataset)
    loaded={t:D.load_modality(dataset,t) for t in tags}
    isbin={t:detect_feature_type(loaded[t][0].to_numpy()) for t in tags}
    all_results={t:[] for t in tags}; coverage={}; t0=time.time()
    for seed in seeds:
        masks={}; splits={}
        for t in tags:
            X,y,cn=loaded[t]; Xtr,Xte,ytr,yte=D.make_split(X,y,seed); splits[t]=(Xtr,Xte,ytr,yte,cn)
            for mk in models:
                clf=train_model(mk,Xtr,ytr,seed=seed)
                masks[f"{t}|{mk}"]=(clf.predict(Xte.to_numpy())==yte.to_numpy())
        common=common_correct_indices(masks)
        coverage[seed]=dict(n_test=int(len(next(iter(masks.values())))),n_common=int(len(common)))
        print(f"[seed {seed}] common-correct: {len(common)}/{coverage[seed]['n_test']}")
        for t in tags:
            Xtr,Xte,ytr,yte,cn=splits[t]
            for mk in models:
                ts=time.time(); cell=run_cell(t,mk,seed,Xtr,Xte,ytr,yte,cn,common,isbin[t],n_shap,n_lime)
                all_results[t].append(cell)
                print(f"  [{t}|{mk}|s{seed}] F1={cell['macro_f1']:.3f} n={cell['n_explained']} ({time.time()-ts:.0f}s)")
    for t in tags:
        cn=loaded[t][2]
        json.dump(dict(tag=t,dataset=dataset,class_names=cn,coverage=coverage,cells=all_results[t]),
                  open(os.path.join(out_dir,f"{dataset}_{t}_fixed.json"),"w"))
    print(f"TOTAL {time.time()-t0:.0f}s")
    _aggregate(dataset, all_results, out_dir)
    return all_results


def _ci95(vals):
    vals=np.asarray(vals,float); n=len(vals)
    if n<2: return (float(np.mean(vals)) if n else np.nan,np.nan,np.nan)
    m=vals.mean(); se=vals.std(ddof=1)/np.sqrt(n); h=se*tdist.ppf(0.975,n-1); return float(m),float(m-h),float(m+h)

def _aggregate(dataset, all_results, out_dir):
    CORE=["del_median","ins_median","stab_adjjac","stab_rbo","stab_rho","seed_adjjac",
          "intr_shap_adjjac","shap_lime_adjjac_disc1","shap_lime_adjjac_disc0"]
    def seed_means(tag,model,metric):
        out=[]
        for cell in all_results[tag]:
            if cell["model"]!=model: continue
            v=[r[metric] for r in cell["records"] if metric in r]
            if v: out.append(np.mean(v))
        return out
    rows=[]
    for tag in all_results:
        for model in sorted({c["model"] for c in all_results[tag]}):
            f1=[c["macro_f1"] for c in all_results[tag] if c["model"]==model]
            row=dict(dataset=dataset,modality=tag,model=model,macro_f1=round(np.mean(f1),4),
                     n_common=all_results[tag][0].get("n_common"))
            for m in CORE:
                mean,lo,hi=_ci95(seed_means(tag,model,m)); row[m]=round(mean,4)
                row[m+"_ci"]=f"[{lo:.3f},{hi:.3f}]" if not np.isnan(lo) else "NA"
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir,f"{dataset}_aggregate_ci.csv"),index=False)
    # paired significance
    def paired_matrix(model,metric):
        mods=list(all_results); table={}
        for t in mods:
            for c in all_results[t]:
                if c["model"]!=model: continue
                for r in c["records"]:
                    if metric in r: table.setdefault((c["seed"],r["idx"]),{})[t]=r[metric]
        rows=[[v[t] for t in mods] for v in table.values() if all(t in v for t in mods)]
        return np.array(rows),mods
    sig=[]
    for model in sorted({c["model"] for t in all_results for c in all_results[t]}):
        for m in ["stab_adjjac","stab_rbo","stab_rho","del_median","shap_lime_adjjac_disc1","intr_shap_adjjac"]:
            M,_=paired_matrix(model,m)
            if M.shape[0]<3 or M.shape[1]<2: continue
            chi2,p,W=friedman_paired(M)
            sig.append(dict(dataset=dataset,model=model,metric=m,n_pairs=int(M.shape[0]),
                            chi2=chi2,p=p,kendall_w=W,sig=sig_stars(p)))
    pd.DataFrame(sig).to_csv(os.path.join(out_dir,f"{dataset}_paired_significance.csv"),index=False)
    print(f"saved aggregate + paired significance to {out_dir}")
