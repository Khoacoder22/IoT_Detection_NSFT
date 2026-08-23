#!/usr/bin/env python3
"""Build the balanced N-BaIoT CSV from the UCI per-device files.

Mirrors the notebook's Import_dataset + Preprocess_data (cell 5) + Get_sample_ratio_by_column:
    1. Walk 9 devices × 11 files each (benign + 5 mirai subtypes + 5 gafgyt subtypes)
    2. Tag each row with 'Names Atk' (benign → 'Benign', else attack subtype: ack, scan, syn, udp,
       udpplain, combo, junk, tcp)
    3. Subsample each attack subtype to `limit` rows (random_state=42); subsample benign to `limit`
    4. Write to data/N_BaIoT_{limit}.csv with 'Names Atk' as the label column.

Usage: python3 code/build_n_baiot.py [--raw-dir DIR] [--limit 1000] [--output FILE]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEVICES = [
    "Danmini_Doorbell",
    "Ecobee_Thermostat",
    "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor",
    "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera",
    "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
]

_ATK_NAMES = {
    "mirai":  ["ack", "scan", "syn", "udp", "udpplain"],
    "gafgyt": ["combo", "junk", "scan", "tcp", "udp"],
}

SEED = 42


def _find_device_dir(raw_dir: Path, device_name: str) -> Path | None:
    """UCI extraction can produce either flat files ({id}.benign.csv) or per-device subdirs
    ({device_name}/benign_traffic.csv, {device_name}/mirai_attacks/ack.csv, etc.).
    Return the containing directory to search in."""
    cand = raw_dir / device_name
    if cand.is_dir():
        return cand
    # Flat layout: files are in raw_dir/
    return raw_dir


def _read_benign(raw_dir: Path, device_name: str, device_id: int) -> pd.DataFrame | None:
    """Try a few known filename patterns for benign traffic."""
    candidates = [
        raw_dir / f"{device_id}.benign.csv",               # flat layout
        raw_dir / device_name / "benign_traffic.csv",       # per-device layout
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    return None


def _read_attack(raw_dir: Path, device_name: str, device_id: int, bot: str, atk: str) -> pd.DataFrame | None:
    candidates = [
        raw_dir / f"{device_id}.{bot}.{atk}.csv",               # flat
        raw_dir / device_name / f"{bot}_attacks" / f"{atk}.csv", # per-device
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    return None


def build(raw_dir: Path, limit: int, bot_type: str = "both") -> pd.DataFrame:
    """Build the concat-then-sample DataFrame.

    bot_type: 'both' (default, matches notebook bot_type=2), 'mirai', or 'gafgyt'.
    """
    if bot_type == "mirai":
        bots = ["mirai"]
    elif bot_type == "gafgyt":
        bots = ["gafgyt"]
    else:
        bots = ["mirai", "gafgyt"]

    # --- benign ---
    nors = []
    for idx, device in enumerate(DEVICES, start=1):
        df = _read_benign(raw_dir, device, idx)
        if df is None:
            print(f"[warn] benign missing for device {idx} ({device})")
            continue
        df = df.drop_duplicates(keep=False)
        df["Names Atk"] = "Benign"
        df["Names Bot"] = "Benign"
        df["Devices"] = device
        nors.append(df)
    if not nors:
        raise RuntimeError(f"No benign files found under {raw_dir}")
    df_nors = pd.concat(nors, ignore_index=True).reset_index(drop=True)
    df_nors["Label"] = 0
    print(f"[build] benign rows: {len(df_nors)}")

    # --- attacks ---
    anos = []
    for idx, device in enumerate(DEVICES, start=1):
        device_bots = []
        for bot in bots:
            device_atks = []
            for atk in _ATK_NAMES[bot]:
                df = _read_attack(raw_dir, device, idx, bot, atk)
                if df is None:
                    print(f"[warn] missing {device} {bot}.{atk}")
                    continue
                df = df.drop_duplicates(keep=False)
                df["Names Atk"] = atk
                device_atks.append(df)
            if device_atks:
                part = pd.concat(device_atks, ignore_index=True).reset_index(drop=True)
                part["Names Bot"] = bot
                device_bots.append(part)
        if device_bots:
            dfd = pd.concat(device_bots, ignore_index=True).reset_index(drop=True)
            dfd["Devices"] = device
            anos.append(dfd)
    if not anos:
        raise RuntimeError(f"No attack files found under {raw_dir}")
    df_anos = pd.concat(anos, ignore_index=True).reset_index(drop=True)
    df_anos["Label"] = 1
    print(f"[build] attack rows: {len(df_anos)}  attack subtypes: {sorted(df_anos['Names Atk'].unique())}")

    # --- subsample to `limit` per class (matches Get_sample_ratio_by_column) ---
    def _sample_per(df, col, n):
        parts = []
        for v in df[col].value_counts().index:
            sub = df[df[col] == v]
            take = min(len(sub), n)
            parts.append(sub.sample(n=take, random_state=SEED))
        return pd.concat(parts, ignore_index=True).reset_index(drop=True)

    df_anos = _sample_per(df_anos, "Names Atk", limit)
    df_nors = _sample_per(df_nors, "Label", limit)  # single class (0) → takes `limit` rows

    # --- merge + preprocess (notebook's Preprocess_data) ---
    df = pd.concat([df_anos, df_nors], ignore_index=True).reset_index(drop=True)
    df = df.drop(columns=["Label", "Names Bot", "Devices"], errors="ignore")
    # 'Names Atk' stays as the label column — renamed to 'Label' downstream in datasets.py
    print(f"[build] final rows: {len(df)}  columns: {df.shape[1]}  class counts: {df['Names Atk'].value_counts().to_dict()}")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/N_BaIoT", help="Directory with extracted UCI files")
    ap.add_argument("--limit", type=int, default=1000, help="Rows per class (default 1000 matches paper)")
    ap.add_argument("--output", default=None, help="Output CSV path (default data/N_BaIoT_{limit}.csv)")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    if not raw.exists():
        print(f"[error] raw directory {raw} does not exist", file=sys.stderr)
        return 2

    df = build(raw, args.limit)
    out = Path(args.output) if args.output else Path("data") / f"N_BaIoT_{args.limit}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[build] wrote {out} ({out.stat().st_size/1024/1024:.1f} MB, {len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
