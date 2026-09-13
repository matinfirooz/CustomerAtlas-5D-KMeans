#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_pipeline import fetch_uci_online_retail

if __name__ == "__main__":
    df = fetch_uci_online_retail(ROOT / "data" / "online_retail.csv")
    print(df.shape)
    print(df.head())
    print("Saved:", ROOT / "data" / "online_retail.csv")
