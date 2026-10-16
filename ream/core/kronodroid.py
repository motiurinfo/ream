"""KronoDroid loader (dhoogla parquet version). Time-based hybrid dataset.

Two device splits: real (kronodroid-2021-real-v1.parquet) and emulator
(kronodroid-2021-emu-v1.parquet). Each row carries BOTH dynamic (system-call) and
static (permission + manifest-structure) features on the SAME sample -> a genuine
row-aligned static-vs-dynamic cross-modal dataset (unlike CCCS). Plus per-sample
timestamps -> enables the temporal / concept-drift study (Contribution 4) and a
cross-device robustness study (real vs emulator).

Modalities (by column-name pattern):
  K1_Syscall : dynamic system-call features (the leading numeric columns)
  K2_Perm    : static Android permissions (ALL_CAPS names)
  K3_Static  : static manifest-structure counts (Activities, NrServices, intents...)

Meta columns (Package, Malware, MalFamily, dates, Scanners, Detection_Ratio, ...)
are excluded from features. Label = binary `Malware` by default (or `MalFamily`).
Timestamp = HighestModDate (real) / LastModDate (emu) -> parsed to a year.
"""
import os
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from . import config as C

META_COLS = {"Package","Malware","MalFamily","EarliestModDate","HighestModDate",
             "FirstModDate","LastModDate","TimesSubmitted","Scanners","Detection_Ratio",
             "FilesInsideAPK","sha256","Hash","nr_syscalls"}

# static manifest-structure count columns (extend if the file has more)
STATIC_STRUCT = {"Activities","NrIntServices","NrIntServicesActions","NrIntActivities",
    "NrIntActivitiesActions","NrIntReceivers","NrIntReceiversActions","TotalIntentFilters",
    "NrServices","NrActivities","NrReceivers","NrProviders"}

KRONO_FILES = {"real":"kronodroid-2021-real-v1.parquet",
               "emu": "kronodroid-2021-emu-v1.parquet"}
KRONO_TS    = {"real":"HighestModDate", "emu":"LastModDate"}
KRONO_MODALITIES = ["K1_Syscall","K2_Perm","K3_Static"]


def _classify(col):
    if col in META_COLS: return "meta"
    if col in STATIC_STRUCT: return "K3_Static"
    if col.isupper() and ("_" in col or len(col) > 3): return "K2_Perm"
    return "K1_Syscall"


def _load_full(device="real"):
    path = os.path.join(C.DATA_DIR, KRONO_FILES[device])
    if not os.path.exists(path):
        alt = os.path.join(C.DATA_DIR, "**", KRONO_FILES[device])
        import glob; hits = glob.glob(alt, recursive=True)
        if hits: path = hits[0]
        else: raise FileNotFoundError(f"{KRONO_FILES[device]} not found under {C.DATA_DIR}")
    df = pd.read_parquet(path)
    # categorical -> plain
    for c in df.columns:
        if str(df[c].dtype).startswith("category"):
            df[c] = df[c].astype(str)
    return df


def _parse_year(series):
    """KronoDroid dates are M/D/YYYY strings; return integer year, NaN if invalid."""
    dt = pd.to_datetime(series, errors="coerce", format="%m/%d/%Y")
    yr = dt.dt.year
    # drop absurd sentinel years (1980/1981 are known invalid placeholders)
    yr = yr.where((yr >= 2008) & (yr <= 2020))
    return yr


def load_kronodroid(tag, device="real", label="Malware"):
    """Return (X, y, class_names) for one modality tag of KronoDroid."""
    df = _load_full(device)
    cols_by_group = {}
    for c in df.columns:
        cols_by_group.setdefault(_classify(c), []).append(c)
    feat_cols = cols_by_group.get(tag, [])
    if not feat_cols:
        raise ValueError(f"no columns for modality {tag}; found groups {list(cols_by_group)}")
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    X = X.loc[:, X.sum(axis=0) > 0].reset_index(drop=True)

    if label == "Malware":
        y = df["Malware"].astype(int).reset_index(drop=True)
        class_names = ["Benign","Malware"]
    else:  # family (many classes; drop tiny)
        fam = df["MalFamily"].astype(str)
        vc = fam.value_counts(); keep = vc[vc >= C.CCCS_MINCLASS].index
        mask = fam.isin(keep)
        df = df[mask]; X = X[mask.values]
        present = sorted(fam[mask].unique()); remap = {c:i for i,c in enumerate(present)}
        y = fam[mask].map(remap).reset_index(drop=True); class_names = present
        X = X.reset_index(drop=True)
    print(f"[{tag}/{device}] X={X.shape}, label={label}, classes={len(class_names)}")
    return X, y, class_names


def load_kronodroid_with_year(device="real", label="Malware"):
    """Full frame + all three modality column groups + parsed year, for temporal study.
    Returns (df_features_dict, y, year_series). df_features_dict maps modality->X."""
    df = _load_full(device)
    year = _parse_year(df[KRONO_TS[device]])
    cols_by_group = {}
    for c in df.columns:
        cols_by_group.setdefault(_classify(c), []).append(c)
    mods = {}
    for tag in KRONO_MODALITIES:
        cols = cols_by_group.get(tag, [])
        X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        mods[tag] = X.loc[:, X.sum(axis=0) > 0].reset_index(drop=True)
    y = df["Malware"].astype(int).reset_index(drop=True) if label=="Malware" else None
    return mods, y, year.reset_index(drop=True)


def make_split(X, y, seed):
    return train_test_split(X, y, test_size=C.TEST_SIZE, random_state=seed, stratify=y)
