"""Detector families (LightGBM, RF, MLP) + intrinsic logistic reference."""
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from . import config as C


def train_model(kind, X_train, y_train, seed=C.SEED):
    Xn,yn=np.asarray(X_train),np.asarray(y_train)
    n_classes=len(np.unique(yn))
    if kind=="lgbm":
        objective="binary" if n_classes==2 else "multiclass"
        m=lgb.LGBMClassifier(objective=objective,n_estimators=400,learning_rate=0.05,
            num_leaves=64,subsample=0.9,colsample_bytree=0.9,class_weight="balanced",
            random_state=seed,n_jobs=-1,verbose=-1)
    elif kind=="rf":
        m=RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=seed,n_jobs=-1)
    elif kind=="mlp":
        m=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(128,64),
            max_iter=120,early_stopping=True,random_state=seed))
    else: raise ValueError(kind)
    m.fit(Xn,yn); return m


def train_intrinsic(X_train, y_train, seed=C.SEED):
    m=make_pipeline(StandardScaler(),LogisticRegression(penalty="l2",solver="lbfgs",C=1.0,
        max_iter=300,class_weight="balanced",n_jobs=-1,random_state=seed))
    m.fit(np.asarray(X_train),np.asarray(y_train)); return m

def intrinsic_attributions(clf, x, c):
    sc=clf.named_steps["standardscaler"]; lr=clf.named_steps["logisticregression"]
    xs=sc.transform(x.reshape(1,-1)).reshape(-1)
    coef=lr.coef_
    if coef.shape[0]==1:      # binary: single coef row; class 1 is +, class 0 is -
        w = coef[0] if c==1 else -coef[0]
    else:
        w = coef[c]
    return w*xs


def evaluate(clf, X_test, y_test, cn, name=""):
    yp=clf.predict(np.asarray(X_test))
    yt=np.asarray(y_test)
    return dict(name=name, macro_f1=f1_score(yt,yp,average="macro"),
        weighted_f1=f1_score(yt,yp,average="weighted"),
        confusion=confusion_matrix(yt,yp).tolist(),
        report=classification_report(yt,yp,target_names=cn,digits=4,zero_division=0))
