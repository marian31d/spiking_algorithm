import argparse
from pathlib import Path
from io import StringIO

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
    try:
        return pd.read_csv(StringIO(block), header=0, sep=sep)
    except Exception as e:
        raise RuntimeError(f"Failed to parse block: {e}")


def normalize_label(label: str) -> str:
    return " ".join(str(label).strip().lower().split())


def is_template_similarity_label(label: str) -> bool:
    s = normalize_label(label)
    return s in {"ts mean", "ts spk-wt", "ts sqt-wt"}


def plot_single_dataset(dataset_name, metric_labels, values, title, ylabel):
    x = np.arange(len(metric_labels))

    plt.figure(figsize=(9, 5))
    plt.plot(
        x,
        values,
        marker="o",
        linestyle="-",
        markersize=5
    )

    plt.xticks(x, metric_labels, rotation=30, ha="right")
    plt.xlabel("Metric")
    plt.ylabel(ylabel)
    plt.title(f"{title} - {dataset_name}")
    plt.grid(True, alpha=0.3)

    finite_vals = np.asarray(values, dtype=float)
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    if finite_vals.size > 0:
        ymin = np.nanmin(finite_vals)
        ymax = np.nanmax(finite_vals)
        margin = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
        plt.ylim(ymin - margin, ymax + margin)

    plt.tight_layout()
    plt.show()


def plot_metric_table_per_dataset(df: pd.DataFrame, direction_title: str):
    metric_labels = df.iloc[:, 0].astype(str).tolist()
    dataset_names = df.columns[1:].tolist()
    values = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")

    for col_idx, dataset_name in enumerate(dataset_names):
        dataset_values = values.iloc[:, col_idx].to_numpy(dtype=float)

        non_ts_labels = []
        non_ts_values = []

        ts_labels = []
        ts_values = []

        for lbl, val in zip(metric_labels, dataset_values):
            if is_template_similarity_label(lbl):
                ts_labels.append(lbl)
                ts_values.append(val)
            else:
                non_ts_labels.append(lbl)
                non_ts_values.append(val)

        if len(non_ts_labels) > 0:
            plot_single_dataset(
                dataset_name=dataset_name,
                metric_labels=non_ts_labels,
                values=non_ts_values,
                title=direction_title,
                ylabel="Score (%)",
            )

        if len(ts_labels) > 0:
            plot_single_dataset(
                dataset_name=dataset_name,
                metric_labels=ts_labels,
                values=ts_values,
                title=f"{direction_title} - Template Similarity",
                ylabel="Template similarity",
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp_file", required=True, type=str)
    ap.add_argument("--sep", default=",")
    args = ap.parse_args()

    csv_file = Path(args.comp_file).expanduser().resolve()
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    blocks = split_tables(csv_file)
    print(f"Found {len(blocks)} table block(s).")

    direction_titles = [
        "GUI→FLOW",
        "FLOW→GUI",
    ]

    for i, block in enumerate(blocks):
        df = parse_table_block(block, sep=args.sep)
        title = direction_titles[i] if i < len(direction_titles) else f"Table {i+1}"
        plot_metric_table_per_dataset(df, title)


if __name__ == "__main__":
    main()
