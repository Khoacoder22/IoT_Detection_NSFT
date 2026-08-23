"""Dataset loaders. Reads the balanced CSVs under data/ and applies the per-dataset
label-column selection + category remap used in the notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .paths import DATA_DIR

DATASETS = ["BoT_IoT", "ToN_IoT", "N_BaIoT", "UNSW_NB15", "CIC_IoT2023", "IoTID20", "Edge_IIoTset", "5G_NIDD"]

_DATASET_FILES = {
    ("BoT_IoT", 1000):     "BoT_IoT_1000.csv",
    ("CIC_IoT2023", 1000): "CIC_IoT2023_1000.csv",
    ("ToN_IoT", 1000):     "ToN_IoT_1000.csv",
    ("ToN_IoT", 2000):     "ToN_IoT_2000.csv",
    ("UNSW_NB15", 1000):   "UNSW_NB15_1000.csv",
    ("IoTID20", 1000):     "iotid20_1000.csv",
    ("IoTID20", 2000):     "iotid20_2000.csv",
    ("N_BaIoT", 1000):     "N_BaIoT_1000.csv",  # produced by code/build_n_baiot.py
    ("Edge_IIoTset", 1000): "edge_iiotset_1000.csv",  # built from Papers/cnNFST raw (15 classes × 1000)
    ("5G_NIDD", 1000):      "5G_NIDD_1000.csv",       # built from Papers/cnNFST raw (9 classes × 1000)
}

_DEFAULT_LIMIT = {
    "BoT_IoT":     1000,
    "CIC_IoT2023": 1000,
    "ToN_IoT":     1000,
    "UNSW_NB15":   1000,
    "IoTID20":     2000,
    "N_BaIoT":     1000,
    "Edge_IIoTset": 1000,
    "5G_NIDD":     1000,
}

# Which column in the raw CSV holds the label the notebook actually trains on.
_LABEL_COLUMN = {
    "BoT_IoT":     "subcategory",  # notebook maps subcategory → 6 classes
    "CIC_IoT2023": "Label",
    "ToN_IoT":     "type",
    "UNSW_NB15":   "attack_cat",
    "IoTID20":     "Target",       # will be renamed to Label
    "N_BaIoT":     "Names Atk",  # produced by build_n_baiot.py; 9 classes incl. Benign
    "Edge_IIoTset": "Attack_type",  # 15 classes including Normal
    "5G_NIDD":      "Attack_Type",  # 9 classes including Benign (5G network attacks)
}

# notebook _category_map (cell 6) — keep only these keys, map their values.
_CATEGORY_MAP = {
    "BoT_IoT": {
        "0Normal":           "0Normal",
        "Data_Exfiltration": "theft",
        "HTTP":              "HTTP",
        "Keylogging":        "theft",
        "OS_Fingerprint":    "scan",
        "Service_Scan":      "scan",
        "TCP":               "TCP",
        "UDP":               "UDP",
    },
    "ToN_IoT": {
        "0Normal":    "0Normal",
        "backdoor":   "Malware",
        "ransomware": "Malware",
        "scanning":   "Scan",
        "password":   "BruteForce",
        "ddos":       "DDoS",
        "xss":        "WebAttack",
        "injection":  "WebAttack",
        "dos":        "DoS",
        "mitm":       "MITM",
    },
    "UNSW_NB15": {
        # Notebook cell-6 has a duplicate-key dict:
        #     'DoS': 'DoS', 'DoS': 'Fuzzers'
        # Python keeps the last assignment, so effectively DoS → Fuzzers, and raw 'Fuzzers'
        # + 'Generic' values are filtered out (not present as keys).
        # Port the notebook's resulting 8-class scheme exactly.
        "0Normal":        "0Normal",
        "Exploits":       "Exploits",
        "Shellcode":      "Shellcode",
        "Backdoor":       "Backdoor",
        "Worms":          "Worms",
        "DoS":            "Fuzzers",
        "Reconnaissance": "Reconnaissance",
        "Analysis":       "Analysis",
    },
    "CIC_IoT2023": {
        "0Normal":          "0Normal",
        "DDoS-UDP_Flood":   "DDoS/DoS",
        "DDoS-ICMP_Flood":  "DDoS/DoS",
        "DDoS-TCP_Flood":   "DDoS/DoS",
        "MITM-ArpSpoofing": "Spoofing",
        "DoS-TCP_Flood":    "DDoS/DoS",
        "DoS-UDP_Flood":    "DDoS/DoS",
        "VulnerabilityScan": "Scan",
        "Backdoor_Malware": "Web",
        "Mirai-udpplain":   "Mirai",
    },
    # IoTID20 and N_BaIoT: no remap — use labels as-is (per notebook load_ids_by_name).
}

# Columns to drop unconditionally — alt-labels, identifiers, high-cardinality strings.
_DROP_ALWAYS = {
    "attack", "category", "subcategory",        # BoT_IoT alt-labels (remove after picking label)
    "type", "Label", "label", "Target",          # alt-label candidates
    "saddr", "daddr", "sport", "dport",          # network addresses as strings
    "src_ip", "dst_ip", "srcip", "dstip",
    "Binary_dtloader", "Category_dtloader",
    # Capitalised alt-labels: BoT_IoT and CIC_IoT2023 ship a coarse "Category"
    # alongside the lowercase one, and CIC_IoT2023 also ships a binary
    # normal/attack flag. Both are derived from the label, so keeping them as
    # features leaks the target (CIC_IoT2023 "Category" maps 1:1 onto the
    # 6 remapped classes).
    "Category", "Binary",
    "Flow_ID", "Src_IP", "Dst_IP", "Timestamp",  # IoTID20 identifiers
    "attack_cat",                                # UNSW alt-label
}

_BENIGN_LABELS = {"BENIGN", "Normal", "normal", "Benign", "BenignTraffic", "Normal_Normal"}


def _clean(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Final cleanup after label column is in place as 'Label'."""
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    # Drop alt-labels / identifiers except the chosen label column
    for c in _DROP_ALWAYS:
        if c in df.columns and c != label_col:
            df = df.drop(columns=[c])

    # Label-encode remaining object columns (except the label itself)
    for c in df.select_dtypes(include="object").columns:
        if c == label_col:
            continue
        df[c] = df[c].astype("category").cat.codes

    # Drop all-NaN feature columns (e.g., BoT_IoT smac/dmac/soui/doui/sco/dco)
    all_nan = [c for c in df.columns if c != label_col and df[c].isna().all()]
    if all_nan:
        df = df.drop(columns=all_nan)

    # Ensure label is last column
    if label_col in df.columns and df.columns[-1] != label_col:
        cols = [c for c in df.columns if c != label_col] + [label_col]
        df = df[cols]
    return df


def load_dataset(name: str, limit: int | None = None) -> tuple[pd.DataFrame, int]:
    """Load a dataset by name. Applies notebook's label selection + category map.

    Returns (dataframe, limit). The last column is always 'Label'. Other columns
    are numeric (categoricals are label-encoded; IP-like/id columns are dropped).
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Valid: {DATASETS}")
    limit = limit or _DEFAULT_LIMIT[name]
    key = (name, limit)
    if key not in _DATASET_FILES:
        raise FileNotFoundError(f"No file for {name} at limit={limit}. Available: {sorted(k for k in _DATASET_FILES if k[0] == name)}")
    path = DATA_DIR / _DATASET_FILES[key]
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    df = pd.read_csv(path, low_memory=False)

    raw_label = _LABEL_COLUMN[name]
    if raw_label not in df.columns:
        raise ValueError(f"{name}: expected label column '{raw_label}' not found; got {list(df.columns)[:10]}...")

    # If raw_label != "Label" and a different 'Label' column already exists, drop it first
    # to avoid a duplicate column name after the rename.
    if raw_label != "Label" and "Label" in df.columns:
        df = df.drop(columns=["Label"])

    # Rename to 'Label'
    df = df.rename(columns={raw_label: "Label"})

    # Normalize benign
    df["Label"] = df["Label"].apply(lambda v: "0Normal" if v in _BENIGN_LABELS else v)

    # Apply notebook's per-dataset category filter + remap
    cat_map = _CATEGORY_MAP.get(name)
    if cat_map is not None:
        df = df[df["Label"].isin(cat_map.keys())].copy()
        df["Label"] = df["Label"].map(cat_map)

    df = _clean(df, label_col="Label")
    return df, limit


def split_features_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Split last column as labels, rest as features."""
    arr = df.to_numpy()
    return arr[:, :-1], arr[:, -1]
