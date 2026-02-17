import argparse
from pathlib import Path
from io import StringIO
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def split_tables(csv_path):
    text = Path(csv_path).read_text().replace("\r\n", "\n")
    lines = text.split("\n")

    blocks = []
    curr = []

    for line in lines:
        if line.strip() == "" or set(line.strip()) == {","}:
            if curr:
                blocks.append("\n".join(curr))
                curr = []
        else:
            curr.append(line)

    if curr:
        blocks.append("\n".join(curr))

    return blocks


def parse_table_block(block: str, sep: str = ",") -> pd.DataFrame:
    """
    Read a table block as raw cells (no header), so the first row can be used as x-axis.
    """
    try:
        return pd.read_csv(StringIO(block), header=None, sep=sep)
    except Exception as e:
        raise RuntimeError(f"Failed to parse a table block as CSV (sep='{sep}'): {e}")


def plot_table(df: pd.DataFrame, title: str, duration: float):
    """
    - First row = x axis (excluding first column)
    - Each next row = one line
      - first column = label
      - remaining columns = y values (NO conversion)
    - Y axis forced from 0 to 100
    """

    # X axis from first row (skip first column)
    # Original x from first row
    x = pd.to_numeric(df.iloc[0, 1:], errors="coerce").to_numpy()

    # Normalize to user duration
    if len(x) > 0:
        x = x / duration


    # Labels from first column (excluding header row)
    labels = df.iloc[1:, 0].astype(str).tolist()

    # Y values (leave exactly as in table, just convert to numeric)
    y_df = df.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce")

    # Make sure lengths match
    n = min(len(x), y_df.shape[1])
    x = x[:n]

    plt.figure(figsize=(10, 6))

    for lbl, row in zip(labels, y_df.values):
        plt.plot(x, row, label=lbl)

    plt.xlabel("X (from first row)")
    plt.ylabel("Percentage (%)")
    plt.title(title)
    plt.grid(True, alpha=0.3)


    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.show()



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp_file", required=True, type=str, help="CSV file containing tables separated by blank lines")
    ap.add_argument("--duration", type=float, required=True, help="Total time duration to normalize x-axis to")
    ap.add_argument("--sep", default=",", help="CSV separator (default: ',')")
    args = ap.parse_args()

    csv_file = Path(args.comp_file).expanduser().resolve()
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    print(f"Reading csv: {csv_file}")
    blocks = split_tables(csv_file)
    print(f"Found {len(blocks)} table block(s) separated by blank lines.")

    for i, block in enumerate(blocks, start=1):
        df = parse_table_block(block, sep=args.sep)
        plot_table(df, title=f"Table {i}", duration=args.duration)


if __name__ == "__main__":
    main()
