import argparse
from pathlib import Path
from io import StringIO
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

"""
Plot comparison tables from a CSV file.

The CSV may contain multiple tables separated by blank or comma-only lines.
Each table is plotted as a separate figure.

For each table:
- The first row defines the x-axis values (excluding the first column).
- The remaining rows are plotted as curves.
- The first column of each row is used as the curve label.
- The y-axis is fixed from 0 to 100 (percentage scale).
- The x-axis is normalized proportionally to a user-provided duration.

Used to visualize percentage metrics over normalized time.
"""

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


def plot_table(df: pd.DataFrame, title: str, xtitle: str, ytitle: str, duration: float, name: str):
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

    fig, ax = plt.subplots(figsize=(10, 6))

    for lbl, row in zip(labels, y_df.values):
        ax.plot(x, row, marker='o',ms = 4, label=lbl)

    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    ax.set_title(title)
    ax.grid(True, color='#d8dcd6')

    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    plt.savefig(name + ".png", dpi=300)
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

    titles = [["Learning time", "Time (seconds)"," Percentage of matched spikes (%)", "time-%"], ["Learning time", "Time (seconds)"," Similarity score of matched spikes (%)", "time-simscore"]]
    for i, block in enumerate(blocks, start=0):
        df = parse_table_block(block, sep=args.sep)
        plot_table(df, title=titles[i][0], xtitle=titles[i][1], ytitle=titles[i][2], duration=args.duration, name=titles[i][3])


if __name__ == "__main__":
    main()
