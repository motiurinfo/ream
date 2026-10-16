"""Corrected reliability metrics (reviewer fixes W1-W4).

W1 modality-aware perturbation | W2 chance-corrected agreement |
W3 paired Friedman + Kendall's W | W4 common-correct sampling.
"""
import numpy as np
from scipy.stats import spearmanr, friedmanchisquare


# ---- W1: modality-aware, validity-preserving perturbation ----
def detect_feature_type(X_train_np, thresh_unique=3):
    isb=np.zeros(X_train_np.shape[1],dtype=bool)
    for j in range(X_train_np.shape[1]):
        u=np.unique(X_train_np[:,j])
        if set(u.tolist())<={0.0,1.0} or (len(u)<=thresh_unique and u.max()<=1.0): isb[j]=True
    return isb

def perturb_instance(x, is_binary, rng, frac=0.05):
    xp=x.copy()
    cnt=np.where(~is_binary)[0]; act=cnt[x[cnt]>0]
    if len(act):
        m=max(1,int(round(frac*len(act)))); sel=rng.choice(act,size=m,replace=False)
        xp[sel]=np.maximum(0.0,x[sel]*rng.lognormal(0.0,0.25,size=m))
    bi=np.where(is_binary)[0]
    if len(bi):
        ones=bi[x[bi]==1]; zeros=bi[x[bi]==0]
        nf=max(1,int(round(frac*len(ones)))) if len(ones) else 0
        if nf and len(ones): xp[rng.choice(ones,size=min(nf,len(ones)),replace=False)]=0
        if nf and len(zeros): xp[rng.choice(zeros,size=min(nf,len(zeros)),replace=False)]=1
    return xp


# ---- W2: chance-corrected + dimensionality-robust agreement ----
def topk_jaccard(a,b,k=10):
    A=set(np.argsort(-np.abs(a))[:k]); B=set(np.argsort(-np.abs(b))[:k]); return len(A&B)/len(A|B)
def expected_random_jaccard(k,d):
    inter=(k*k)/d; union=2*k-inter; return inter/union if union>0 else 0.0
def adjusted_jaccard(a,b,k=10,d=None):
    if d is None: d=len(a)
    J=topk_jaccard(a,b,k); EJ=expected_random_jaccard(k,d)
    return (J-EJ)/(1-EJ) if (1-EJ)>0 else J
def rbo(a,b,k=50,p=0.9):
    ra=list(np.argsort(-np.abs(a)))[:k]; rb=list(np.argsort(-np.abs(b)))[:k]
    sa,sb=set(),set(); ov=0.0; w=(1-p)
    for i in range(max(len(ra),len(rb))):
        if i<len(ra): sa.add(ra[i])
        if i<len(rb): sb.add(rb[i])
        ov+=w*(len(sa&sb)/(i+1)); w*=p
    return ov

def perturbation_stability(explain_fn, x, is_binary, d, n_reps=8, frac=0.05, k=10, seed=0):
    rng=np.random.default_rng(seed); base=explain_fn(x)
    jadj,rbos,rho=[],[],[]
    for _ in range(n_reps):
        a=explain_fn(perturb_instance(x,is_binary,rng,frac=frac))
        jadj.append(adjusted_jaccard(base,a,k=k,d=d)); rbos.append(rbo(base,a,k=50))
        rho.append(spearmanr(np.abs(base),np.abs(a)).correlation)
    return float(np.mean(jadj)),float(np.mean(rbos)),float(np.nanmean(rho))


# ---- faithfulness (multi-baseline) ----
def _p(clf,X,c): return clf.predict_proba(X)[:,c]
def make_baselines(Xtr,ytr,kinds,rng):
    b={}
    if "median" in kinds: b["median"]=np.median(Xtr,axis=0)
    if "zero" in kinds: b["zero"]=np.zeros(Xtr.shape[1])
    if "classmean" in kinds: b["classmean"]=None
    if "random" in kinds: b["random"]="__random__"
    return b
def resolve_baseline(name,store,Xtr,ytr,c,rng):
    if name=="classmean": return Xtr[ytr==c].mean(axis=0)
    if name=="random": return Xtr[rng.integers(0,len(Xtr))]
    return store[name]
def deletion_insertion_auc(clf,x,attr,c,baseline,n_steps=20):
    order=np.argsort(-np.abs(attr)); n=len(order); step=max(1,n//n_steps)
    idx=list(range(0,n+1,step));  idx=idx+[n] if idx[-1]!=n else idx
    xd=x.copy(); dp=[_p(clf,xd.reshape(1,-1),c)[0]]
    for i in range(1,len(idx)):
        xd[order[idx[i-1]:idx[i]]]=baseline[order[idx[i-1]:idx[i]]]; dp.append(_p(clf,xd.reshape(1,-1),c)[0])
    xi=baseline.copy(); ip=[_p(clf,xi.reshape(1,-1),c)[0]]
    for i in range(1,len(idx)):
        f=order[idx[i-1]:idx[i]]; xi[f]=x[f]; ip.append(_p(clf,xi.reshape(1,-1),c)[0])
    xs=np.array(idx)/n; return float(np.trapezoid(dp,xs)), float(np.trapezoid(ip,xs))


# ---- W3: paired Friedman + Kendall's W ----
def friedman_paired(matrix):
    groups=[matrix[:,j] for j in range(matrix.shape[1])]
    chi2,p=friedmanchisquare(*groups); n,k=matrix.shape
    return float(chi2),float(p),float(chi2/(n*(k-1)))

def sig_stars(p):
    return "***" if p<1e-3 else ("**" if p<1e-2 else ("*" if p<0.05 else "ns"))


# ---- W4: common-correct selection ----
def common_correct_indices(masks):
    return np.where(np.logical_and.reduce(list(masks.values())))[0]
