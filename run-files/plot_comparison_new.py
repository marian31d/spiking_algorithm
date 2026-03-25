import argparse
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Plot comparison tables from a CSV file.

The CSV may contain multiple tables separated by blank or comma-only lines.
Each table is plotted as one or two figures:

- Main figure: all non-template-similarity rows
- Separate figure: template-similarity rows only

For each table:
- The first row defines the x-axis values (excluding the first column).
- The remaining rows are plotted as curves.
- The first column of each row is used as the curve label.
- The x-axis is normalized proportionally to a user-provided duration.
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


def is_template_similarity_label(label: str) -> bool:
    s = str(label).strip().lower()
    return ("template" in s and "similarity" in s) or ("template" in s)


def prepare_plot_data(df: pd.DataFrame, duration: float):
    # X axis from first row (skip first column)
    x = pd.to_numeric(df.iloc[0, 1:], errors="coerce").to_numpy()

    # Normalize to user duration
    if len(x) > 0:
        x = x / duration

    labels = df.iloc[1:, 0].astype(str).tolist()
    y_df = df.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce")

    # Keep matching length
    n = min(len(x), y_df.shape[1])
    x = x[:n]
    y_df = y_df.iloc[:, :n]

    return x, labels, y_df


def plot_rows(x, labels, y_values, title, ylabel, force_percent_axis=True):
    if len(labels) == 0:
        return

    plt.figure(figsize=(10, 6))

    for lbl, row in zip(labels, y_values):
        plt.plot(
            x,
            row,
            marker='o',          # ← dots
            linestyle='-',       # ← keep lines
            markersize=4,
            label=lbl
        )

    plt.xlabel("Normalized time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if force_percent_axis:
        plt.ylim(0, 100)
    else:
        finite_vals = np.asarray(y_values, dtype=float)
        finite_vals = finite_vals[np.isfinite(finite_vals)]
        if finite_vals.size > 0:
            ymax = np.nanmax(finite_vals)
            ymin = np.nanmin(finite_vals)

            if ymax <= 1.05 and ymin >= -0.05:
                plt.ylim(0, 1)
            else:
                plt.ylim(0, 100)

    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.show()


def plot_table(df: pd.DataFrame, title: str, duration: float):
    x, labels, y_df = prepare_plot_data(df, duration)

    non_template_labels = []
    non_template_rows = []

    template_labels = []
    template_rows = []

    for lbl, row in zip(labels, y_df.values):
        if is_template_similarity_label(lbl):
            template_labels.append(lbl)
            template_rows.append(row)
        else:
            non_template_labels.append(lbl)
            non_template_rows.append(row)

    # Main figure: non-template metrics
    plot_rows(
        x,
        non_template_labels,
        non_template_rows,
        title=title,
        ylabel="Percentage (%)",
        force_percent_axis=True,
    )

    # Separate figure: template similarity only
    if len(template_labels) > 0:
        plot_rows(
            x,
            template_labels,
            template_rows,
            title=f"{title} - Template Similarity",
            ylabel="Template similarity",
            force_percent_axis=False,
        )


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
