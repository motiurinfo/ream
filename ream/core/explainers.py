"""SHAP (model-appropriate) + LIME (both discretizations), common interface."""
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
from . import config as C


class ShapExplainer:
    def __init__(self, clf, kind, X_background=None):
        self.kind=kind
        if kind in ("lgbm","rf"):
            self.explainer=shap.TreeExplainer(clf); self.mode="tree"
        else:
            bg=shap.sample(np.asarray(X_background),50,random_state=C.SEED)
            self.explainer=shap.KernelExplainer(clf.predict_proba,bg); self.mode="kernel"
    def attributions(self, x, c):
        if self.mode=="tree":
            sv=self.explainer.shap_values(x.reshape(1,-1), check_additivity=False)
            if isinstance(sv,list):
                # binary TreeSHAP may return a single array or [neg,pos]
                if len(sv)==1: return np.asarray(sv[0]).reshape(-1)
                return np.asarray(sv[c if c < len(sv) else -1]).reshape(-1)
            sv=np.asarray(sv)
            if sv.ndim==3:  # (n, features, classes)
                idx = c if c < sv.shape[2] else sv.shape[2]-1
                return sv[0,:,idx]
            return sv.reshape(-1)  # binary: (n, features)
        sv=self.explainer.shap_values(x.reshape(1,-1),nsamples=100,silent=True, check_additivity=False)
        return np.asarray(sv[c]).reshape(-1) if isinstance(sv,list) else np.asarray(sv)[0,:,c]


class LimeExplainer:
    def __init__(self, clf, X_train, feat, cn, discretize, seed=C.SEED):
        self.clf=clf; self.n=X_train.shape[1]
        self.explainer=LimeTabularExplainer(np.asarray(X_train),feature_names=list(feat),
            class_names=list(cn),discretize_continuous=discretize,random_state=seed,mode="classification")
    def attributions(self, x, c):
        e=self.explainer.explain_instance(x,self.clf.predict_proba,num_features=self.n,labels=(c,))
        v=np.zeros(self.n)
        for fi,w in e.local_exp[c]: v[fi]=w
        return v
