"""Data loaders for MalDroid-2020 and CCCS-2020 (dynamic-only). Both return
(X DataFrame, y Series, class_names) and share the same downstream interface."""
import os, glob
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from . import config as C

LABEL_CANDIDATES = ["Class","class","Label","label","Category","category","y"]


def _find_label_col(df):
    for c in LABEL_CANDIDATES:
        if c in df.columns: return c
    return None


# ---------------- MalDroid ----------------
def _labels_from_reference(n):
    ref = pd.read_csv(os.path.join(C.DATA_DIR, C.MALDROID_MODALITIES["M2"]))
    ref.columns = [str(c).strip() for c in ref.columns]
    lab = ref["Class"].astype(int).to_numpy(); assert len(lab)==n; return pd.Series(lab)

def _select_static(X):
    if C.STATIC_SELECT=="none": return X
    if C.STATIC_SELECT=="freq":
        pc=(X>0).sum(axis=0); keep=pc[pc>=C.STATIC_MIN_COUNT].index
        print(f"[M3] freq>= {C.STATIC_MIN_COUNT}: kept {len(keep)}/{X.shape[1]}"); return X[keep]
    keep=X.var(axis=0).sort_values(ascending=False).index[:C.STATIC_TOPK]
    print(f"[M3] top-{C.STATIC_TOPK}: kept {len(keep)}/{X.shape[1]}"); return X[keep]

def load_maldroid(tag):
    df=pd.read_csv(os.path.join(C.DATA_DIR, C.MALDROID_MODALITIES[tag]), low_memory=False)
    df.columns=[str(c).strip() for c in df.columns]
    df=df.loc[:,~df.columns.str.match(r"^Unnamed")]
    lc=_find_label_col(df)
    if lc is not None: y_raw=df[lc]; X=df.drop(columns=[lc])
    else: y_raw=_labels_from_reference(len(df)); X=df
    X=X.select_dtypes(include="number").fillna(0); X=X.loc[:,X.sum(axis=0)>0]
    if tag=="M3": X=_select_static(X)
    y_raw=y_raw.astype(int); present=sorted(y_raw.unique()); remap={o:i for i,o in enumerate(present)}
    y=y_raw.map(remap).reset_index(drop=True); X=X.reset_index(drop=True)
    cn=[C.MALDROID_CLASS_NAMES.get(o,str(o)) for o in present]
    print(f"[{tag}] X={X.shape}, label='{lc or 'BORROWED_FROM_M2'}', classes={cn}")
    return X,y,cn


# ---------------- CCCS (dynamic-only) ----------------
_CCCS_CACHE={}
def _cccs_full():
    if "full" in _CCCS_CACHE: return _CCCS_CACHE["full"]
    files=glob.glob(os.path.join(C.DATA_DIR,"**","*_after_reboot_Cat.csv"),recursive=True)
    if not files:
        raise FileNotFoundError(
            f"No *_after_reboot_Cat.csv under {C.DATA_DIR!r} (recursive). Check path.")
    print(f"[CCCS] found {len(files)} after_reboot files")
    fr=[]
    for f in files:
        cat=os.path.basename(f).replace("_after_reboot_Cat.csv","")
        if cat in C.CCCS_DROP: continue
        df=pd.read_csv(f)
        if "Category" not in df.columns: df["Category"]=cat
        fr.append(df)
    full=pd.concat(fr,ignore_index=True); full.columns=[str(c).strip() for c in full.columns]
    full=full.dropna(subset=["Hash","Category"]).copy(); full["Hash"]=full["Hash"].astype(str).str.lower()
    full=full.drop_duplicates(subset=["Hash"],keep="first")
    fc=[c for c in full.columns if c not in C.CCCS_META]
    full[fc]=full[fc].apply(pd.to_numeric,errors="coerce").fillna(0)
    cnts=full["Category"].value_counts(); full=full[full["Category"].isin(cnts[cnts>=C.CCCS_MINCLASS].index)].reset_index(drop=True)
    _CCCS_CACHE["full"]=full
    print(f"[CCCS] {len(full)} apps; {full['Category'].nunique()} categories")
    return full

def load_cccs(tag):
    full=_cccs_full()
    present=sorted(full["Category"].unique()); remap={c:i for i,c in enumerate(present)}
    y=full["Category"].map(remap).reset_index(drop=True)
    feat=[c for c in full.columns if c not in C.CCCS_META]
    cols=[c for c in feat if any(c.startswith(p) for p in C.CCCS_BLOCKS[tag])]
    X=full[cols].reset_index(drop=True); X=X.loc[:,X.sum(axis=0)>0]
    print(f"[{tag}] X={X.shape}, classes={present}")
    return X,y,present


def load_modality(dataset, tag):
    if dataset=="maldroid": return load_maldroid(tag)
    if dataset=="cccs": return load_cccs(tag)
    if dataset.startswith("krono"):
        from .kronodroid import load_kronodroid
        device = "emu" if dataset.endswith("emu") else "real"
        return load_kronodroid(tag, device=device)
    raise ValueError(dataset)

def modalities_for(dataset):
    if dataset=="maldroid": return list(C.MALDROID_MODALITIES)
    if dataset=="cccs": return list(C.CCCS_BLOCKS)
    if dataset.startswith("krono"):
        from .kronodroid import KRONO_MODALITIES; return list(KRONO_MODALITIES)
    raise ValueError(dataset)

def make_split(X,y,seed):
    return train_test_split(X,y,test_size=C.TEST_SIZE,random_state=seed,stratify=y)
