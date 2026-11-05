import argparse
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

REF_PEELS = 100  # peel count of the reference run, drawn as a dashed line

"""
Plot comparison tables from a CSV file.

The CSV may contain multiple tables separated by blank or comma-only lines.
Each table is plotted as one or two figures:

- Main figure: all non-template-similarity rows
- Separate figure: only template-similarity rows:
    * ts mean
    * ts spk-wt
    * ts sqt-wt

For each table:
- The first row defines the x-axis values (excluding the first column).
- The remaining rows are plotted as curves.
- The first column of each row is used as the curve label.
- The x-axis is normalized proportionally to a user-provided duration.

For the peel-sweep table tagged "coverage" in its last header cell:
- A dashed line marks the value at REF_PEELS peels (the REF_PEELS column if the
  table has one, otherwise 100 since that run is the comparison reference).
- The y-axis is in log scale.
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
    try:
        return pd.read_csv(StringIO(block), header=None, sep=sep)
    except Exception as e:
        raise RuntimeError(f"Failed to parse a table block as CSV (sep='{sep}'): {e}")


def normalize_label(label: str) -> str:
    return " ".join(str(label).strip().lower().split())


# one colour per curve label, kept across all figures of a run, so a dataset has
# the same colour in the coverage and the score plots
_label_colors = {}


def color_for(label: str) -> str:
    key = normalize_label(label)
    if key not in _label_colors:
        _label_colors[key] = f"C{len(_label_colors) % 10}"
    return _label_colors[key]


def is_template_similarity_label(label: str) -> bool:
    s = normalize_label(label)
    return s in {"ts mean", "ts spk-wt", "ts sqt-wt"}


def prepare_plot_data(df: pd.DataFrame, duration: float):
    x = pd.to_numeric(df.iloc[0, 1:], errors="coerce").to_numpy()

    if len(x) > 0:
        x = x / duration

    labels = df.iloc[1:, 0].astype(str).tolist()
    y_df = df.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce")

    n = min(len(x), y_df.shape[1])
    x = x[:n]
    y_df = y_df.iloc[:, :n]

    return x, labels, y_df


def reference_values(df: pd.DataFrame):
    """
    Value of each row at REF_PEELS peels, or None if the table is not a peel sweep.
    """
    if "peel" not in normalize_label(df.iloc[0, 0]):
        return None

    x = pd.to_numeric(df.iloc[0, 1:], errors="coerce").to_numpy()
    y_df = df.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce")

    cols = np.flatnonzero(x == REF_PEELS)
    if cols.size > 0:
        return y_df.iloc[:, cols[0]].to_numpy(dtype=float)

    # no REF_PEELS column: that run is the reference, i.e. 100% against itself
    return np.full(len(y_df), 100.0)


def is_coverage_table(df: pd.DataFrame) -> bool:
    return "coverage" in normalize_label(df.iloc[0, -1])


def find_knee(x, y, x_scale="linear", sensitivity=1.0):
    """Knee of a curve that rises and levels off: Kneedle, offline
    (Satopaa, Albrecht, Irwin & Raghavan, ICDCS Workshops 2011, pp. 166-171).

    1. Normalize x and y to [0, 1], so the result does not depend on units.
    2. Difference curve D = y_n - x_n: how far the curve sits above the diagonal.
    3. The knee candidate is the point where D is largest.
    4. Accept it only if D later falls below D_max - sensitivity * (mean spacing
       of x_n), the paper's threshold. With sensitivity=1 a knee has to stand out
       by about one grid step, so a weak bump from one noisy point is rejected.

    A curve that bends upward (lies below the diagonal) has no knee: returns None,
    instead of picking its lowest point. A falling curve is flipped first, so
    "drops and levels off" works the same way.

    `x_scale="log"` normalizes on log10(x) instead, which is the right choice
    when the x grid is roughly geometric (e.g. peel counts 1,2,4,...,90) -- on a
    linear axis the dense low-x end dominates and drags the knee left.

    Returns (x_knee, y_knee, index) or None if there is no knee.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return None

    order = np.argsort(x)
    x, y = x[order], y[order]

    xs = np.log10(x) if (x_scale == "log" and np.all(x > 0)) else x
    xr = xs[-1] - xs[0]
    yr = y.max() - y.min()
    if xr <= 0 or yr <= 0:
        return None            # vertical or perfectly flat: no knee to find

    yy = -y if y[-1] < y[0] else y   # falling curve: flip so it rises
    xn = (xs - xs[0]) / xr
    yn = (yy - yy.min()) / yr

    d = yn - xn
    i = int(np.argmax(d))
    if d[i] <= 0 or i == 0 or i == len(x) - 1:
        return None            # curve never above the diagonal, or max at an endpoint

    threshold = d[i] - sensitivity * np.mean(np.diff(xn))
    if not np.any(d[i + 1:] < threshold):
        return None            # bump too weak to count as a knee
    return float(x[i]), float(y[i]), i


def plot_rows(x, labels, y_values, title, ylabel, force_percent_axis=True, ref_values=None, log_y=False, save_path=None, xlabel="Number of peels", mark_knee=True, knee_x_scale="linear", log_x=False):
    if len(labels) == 0:
        return

    plt.figure(figsize=(10, 6))

    if ref_values is None:
        ref_values = []
    knees = []
    same_ref = len(set(ref_values)) == 1

    for i, (lbl, row) in enumerate(zip(labels, y_values)):
        # datasets cover different x grids, so drop this row's blanks and draw a
        # continuous line through the points it actually has
        rv = np.asarray(row, dtype=float)
        ok = np.isfinite(rv)
        line, = plt.plot(
            np.asarray(x, dtype=float)[ok],
            rv[ok],
            marker="o",
            linestyle="-",
            markersize=4,
            color=color_for(lbl),
            label=lbl
        )

        if mark_knee:
            knee = find_knee(np.asarray(x, dtype=float)[ok], rv[ok], knee_x_scale)
            if knee is not None:
                kx, ky, _ = knee
                # explicit color, otherwise this extra plot call takes the next
                # colour from the cycle
                plt.plot([kx], [ky], marker="o", markersize=11, color=line.get_color(),
                         markerfacecolor="none", markeredgecolor=line.get_color(),
                         markeredgewidth=2, linestyle="none", zorder=5)
                knees.append((kx, ky, lbl, line.get_color()))
        # dashed line at this row's REF_PEELS value
        if i < len(ref_values) and not same_ref and np.isfinite(ref_values[i]):
            plt.axhline(ref_values[i], color=line.get_color(), linestyle="--", linewidth=1,
                        label=f"{lbl} ({REF_PEELS} peels)")

    if same_ref and np.isfinite(ref_values[0]):
        plt.axhline(ref_values[0], color="k", linestyle="--", linewidth=1, label=f"{REF_PEELS} peels")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if log_x:
        xv = np.asarray(x, dtype=float)
        xv = xv[np.isfinite(xv)]
        if xv.size and np.all(xv > 0):
            ax = plt.gca()
            ax.set_xscale("log")
            # plain numbers at 1-2-5 steps instead of powers of ten
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            print("  (log x requested but x has non-positive values; keeping linear)")

    #  NEW: dynamic y-axis
    finite_vals = np.concatenate([np.asarray(y_values, dtype=float).ravel(),
                                  np.asarray(ref_values, dtype=float).ravel()])
    finite_vals = finite_vals[np.isfinite(finite_vals)]

    if finite_vals.size > 0:
        ymin = np.nanmin(finite_vals)
        ymax = np.nanmax(finite_vals)

        if log_y and ymin > 0:
            plt.yscale("log")
            margin = (ymax / ymin) ** 0.05 if ymax > ymin else 1.1
            plt.ylim(ymin / margin, ymax * margin)

            # plain numbers at 1-2-5 steps instead of powers of ten
            ax = plt.gca()
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            margin = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
            plt.ylim(ymin - margin, ymax + margin)
    
    if knees:
        ax = plt.gca()
        y0, y1 = ax.get_ylim()
        for n, (kx, ky, lbl, col) in enumerate(sorted(knees)):
            ax.annotate(f"knee {lbl}: {kx:g}", xy=(kx, ky),
                        xytext=(8, -12 - 13 * n), textcoords="offset points",
                        fontsize=8, color=col,
                        arrowprops=dict(arrowstyle="-", color=col,
                                        linewidth=0.8, alpha=0.6))
        print("  knees (" + ylabel + "): " +
              ", ".join(f"{l}={kx:g}" for kx, _, l, _ in sorted(knees)))

    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def table_ylabel(df: pd.DataFrame) -> str:
    return "Coverage (%)" if is_coverage_table(df) else "Min-norm (%)"


def plot_table(df: pd.DataFrame, title: str, duration: float, out_prefix: Path,
               xlabel: str = "Number of peels", mark_knee: bool = True, log_x: bool = False,
               log_y: bool = False,
               knee_x_scale: str = "linear"):
    x, labels, y_df = prepare_plot_data(df, duration)
    # dashed REF_PEELS line only on the coverage table
    ref = reference_values(df) if is_coverage_table(df) else None

    non_template_labels = []
    non_template_rows = []
    non_template_ref = []

    template_labels = []
    template_rows = []

    for i, (lbl, row) in enumerate(zip(labels, y_df.values)):
        if is_template_similarity_label(lbl):
            template_labels.append(lbl)
            template_rows.append(row)
        else:
            non_template_labels.append(lbl)
            non_template_rows.append(row)
            if ref is not None:
                non_template_ref.append(ref[i])

    # Main figure: everything except ts rows
    plot_rows(
        x,
        non_template_labels,
        non_template_rows,
        title=title,
        ylabel=table_ylabel(df),
        force_percent_axis=True,
        xlabel=xlabel,
        ref_values=non_template_ref,
        # log axes apply to the coverage table only; the score tables stay linear
        log_y=log_y and is_coverage_table(df),
        log_x=log_x and is_coverage_table(df),
        save_path=Path(f"{out_prefix}.png"),
        # knee only on the coverage table: the score curves are too flat for a
        # chord-distance knee to mean anything
        mark_knee=mark_knee and is_coverage_table(df),
        knee_x_scale=knee_x_scale,
    )

    # Separate figure: ts mean / ts spk-wt / ts sqt-wt only
    if len(template_labels) > 0:
        plot_rows(
            x,
            template_labels,
            template_rows,
            title=f"{title} - Template Similarity",
            ylabel="Template similarity",
            force_percent_axis=False,
            save_path=Path(f"{out_prefix}_template_similarity.png"),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp_file", required=True, type=str, help="CSV file containing tables separated by blank lines")
    ap.add_argument("--duration", type=float, required=True, help="Total time duration to normalize x-axis to")
    ap.add_argument("--sep", default=",", help="CSV separator (default: ',')")
    ap.add_argument("--title_prefix", default="Learning Time",
                    help="Prefix for each figure title; the table's own last header cell is appended")
    ap.add_argument("--xlabel", default="Time (seconds)", help="X-axis label")
    ap.add_argument("--no_knee", action="store_true", help="Do not mark the knee of each curve")
    ap.add_argument("--logx", action="store_true", help="Log-scale the x axis (coverage table only)")
    ap.add_argument("--logy", action="store_true", help="Log-scale the y axis (coverage table only)")
    ap.add_argument("--knee_x_scale", default=None, choices=["linear", "log"],
                    help="Normalize x linearly or on log10(x) when locating the knee. "
                         "Use 'log' for geometric grids such as peel counts.")
    args = ap.parse_args()
    # a knee is a property of the curve as drawn, so locate it on the axes we plot
    if args.knee_x_scale is None:
        args.knee_x_scale = "log" if args.logx else "linear"

    csv_file = Path(args.comp_file).expanduser().resolve()
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    label = ["coverage", "scores", "scores(goodclu)"]
    print(f"Reading csv: {csv_file}")
    blocks = split_tables(csv_file)
    print(f"Found {len(blocks)} table block(s) separated by blank lines.")
    for i, block in enumerate(blocks, start=1):
        df = parse_table_block(block, sep=args.sep)
        # each table names itself by its last header cell (coverage / score (...))
        title = f"{args.title_prefix} - {str(df.iloc[0, -1]).strip()}"
        # figures are saved next to the CSV: <csv name>_table<i>.png
        out_prefix = csv_file.with_name(f"{csv_file.stem}_{label[i-1]}")
        plot_table(df, title=title, duration=args.duration,
                   out_prefix=out_prefix, xlabel=args.xlabel,
                   mark_knee=not args.no_knee, knee_x_scale=args.knee_x_scale,
                   log_x=args.logx, log_y=args.logy)


if __name__ == "__main__":
    main()
