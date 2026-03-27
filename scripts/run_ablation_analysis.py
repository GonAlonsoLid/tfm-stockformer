"""
Ablation study analysis and figure generation script.

Reads results/ablation_results.csv and generates:
  - IC comparison bar chart
  - Metrics heatmap
  - Improvement waterfall chart
  - LaTeX table for thesis
  - Statistical significance tests (Spearman IC t-tests)
"""

import os
import glob
import csv
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(REPO_ROOT, "results", "ablation_results.csv")
FIGURES_DIR = os.path.join(REPO_ROOT, "results", "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ablation_results(csv_path: str) -> list[dict]:
    """Load ablation results CSV and return list of row dicts."""
    rows = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            parsed: dict = {"experiment": row["experiment"], "name": row["name"]}
            for col in ("ic", "accuracy", "mae", "rmse", "mape"):
                val = row.get(col, "").strip()
                if val in ("", "nan", "NaN", "None"):
                    parsed[col] = float("nan")
                else:
                    try:
                        parsed[col] = float(val)
                    except ValueError:
                        parsed[col] = float("nan")
            rows.append(parsed)
    return rows


# ---------------------------------------------------------------------------
# Figure 1: IC comparison bar chart
# ---------------------------------------------------------------------------

def _bar_color(ic: float) -> str:
    if np.isnan(ic) or ic < 0:
        return "red"
    if ic < 0.02:
        return "orange"
    return "green"


def plot_ic_comparison(rows: list[dict], out_path: str) -> None:
    experiments = [r["experiment"] for r in rows]
    ics = [r["ic"] if not np.isnan(r["ic"]) else 0.0 for r in rows]
    colors = [_bar_color(r["ic"]) for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(experiments, ics, color=colors, edgecolor="black", linewidth=0.6)

    # Reference lines
    ax.axhline(y=0, color="black", linewidth=1.0, linestyle="-", label="IC = 0")
    ax.axhline(y=0.02, color="orange", linewidth=1.0, linestyle="--",
               label="IC = 0.02 (minimum useful)")
    ax.axhline(y=0.05, color="green", linewidth=1.0, linestyle="--",
               label="IC = 0.05 (strong)")

    # Value labels on each bar
    for bar, ic in zip(bars, [r["ic"] for r in rows]):
        label = f"{ic:.4f}" if not np.isnan(ic) else "N/A"
        y_pos = bar.get_height() + 0.001 if bar.get_height() >= 0 else bar.get_height() - 0.003
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_pos,
            label,
            ha="center",
            va="bottom" if bar.get_height() >= 0 else "top",
            fontsize=9,
        )

    # Legend for colors
    legend_patches = [
        mpatches.Patch(color="red", label="IC < 0"),
        mpatches.Patch(color="orange", label="0 ≤ IC < 0.02"),
        mpatches.Patch(color="green", label="IC ≥ 0.02"),
    ]
    ax.legend(handles=legend_patches + ax.get_legend_handles_labels()[0][0:0],
              loc="upper right")
    # Rebuild legend with reference lines
    ref_handles, ref_labels = ax.get_legend_handles_labels()
    ax.legend(handles=legend_patches + ref_handles, loc="upper right", fontsize=8)

    ax.set_title("Information Coefficient (IC) by Experiment")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("IC")

    # Tick labels: experiment + short name
    labels = [f"{r['experiment']}\n{r['name'][:20]}" for r in rows]
    ax.set_xticks(range(len(experiments)))
    ax.set_xticklabels(labels, fontsize=7, rotation=15, ha="right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Metrics heatmap
# ---------------------------------------------------------------------------

def plot_metrics_heatmap(rows: list[dict], out_path: str) -> None:
    metrics = ["ic", "accuracy", "mae", "rmse", "mape"]
    exp_labels = [r["experiment"] for r in rows]

    data = np.array([[r[m] for m in metrics] for r in rows], dtype=float)

    # Normalise each column to [0, 1] for display (ignoring NaNs)
    normed = np.full_like(data, np.nan)
    for j in range(data.shape[1]):
        col = data[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            continue
        col_min, col_max = valid.min(), valid.max()
        if col_max > col_min:
            normed[:, j] = (col - col_min) / (col_max - col_min)
        else:
            normed[:, j] = 0.5

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("RdYlGn")
    cmap.set_bad(color="lightgrey")

    im = ax.imshow(normed, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=10)
    ax.set_yticks(range(len(exp_labels)))
    ax.set_yticklabels(exp_labels, fontsize=9)

    # Annotate with raw values
    for i, row in enumerate(rows):
        for j, m in enumerate(metrics):
            val = row[m]
            text = f"{val:.4f}" if not np.isnan(val) else "N/A"
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="black")

    fig.colorbar(im, ax=ax, label="Normalised value (column-wise)")
    ax.set_title("Ablation Study — Metrics Heatmap")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Improvement waterfall (E1→E2→E3→E4→E5)
# ---------------------------------------------------------------------------

WATERFALL_EXPERIMENTS = ["E1", "E2", "E3", "E4", "E5"]
WATERFALL_LABELS = [
    "E1\nMSE baseline",
    "E1→E2\nMSE→Ranking",
    "E2→E3\n+Dynamic Graph",
    "E3→E4\n+Rich Features",
    "E4→E5\n+All Combined",
]


def plot_improvement_waterfall(rows: list[dict], out_path: str) -> None:
    by_exp = {r["experiment"]: r["ic"] for r in rows}

    ics = []
    for exp in WATERFALL_EXPERIMENTS:
        ic = by_exp.get(exp, float("nan"))
        ics.append(ic if not np.isnan(ic) else 0.0)

    if len(ics) < len(WATERFALL_EXPERIMENTS):
        print(f"  WARNING: fewer than {len(WATERFALL_EXPERIMENTS)} waterfall "
              f"experiments found — waterfall may be incomplete.")

    # Bars: first bar is absolute IC of E1, subsequent bars are deltas
    bar_values = [ics[0]] + [ics[i] - ics[i - 1] for i in range(1, len(ics))]
    bottoms = [0.0] + list(np.cumsum(ics[:-1]))

    colors = ["steelblue" if v >= 0 else "salmon" for v in bar_values]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(bar_values)), bar_values, bottom=bottoms,
                  color=colors, edgecolor="black", linewidth=0.6)

    # Cumulative IC line
    cumulative = np.cumsum(bar_values) if bottoms[0] == 0 else ics
    ax.step(range(-1, len(cumulative) + 1),
            [cumulative[0]] + list(cumulative) + [cumulative[-1]],
            where="mid", color="navy", linewidth=1.0, linestyle=":", alpha=0.7)

    # Value labels
    for i, (bv, bot) in enumerate(zip(bar_values, bottoms)):
        sign = "+" if bv > 0 else ""
        label = f"{sign}{bv:.4f}"
        y_mid = bot + bv / 2.0
        ax.text(i, y_mid, label, ha="center", va="center", fontsize=8, color="white",
                fontweight="bold")

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.axhline(y=0.02, color="orange", linewidth=1.0, linestyle="--",
               label="IC = 0.02 (minimum useful)")
    ax.axhline(y=0.05, color="green", linewidth=1.0, linestyle="--",
               label="IC = 0.05 (strong)")

    ax.set_xticks(range(len(WATERFALL_LABELS)))
    ax.set_xticklabels(WATERFALL_LABELS, fontsize=8)
    ax.set_ylabel("IC")
    ax.set_title("Incremental IC Improvement (Waterfall)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: LaTeX table
# ---------------------------------------------------------------------------

_METRIC_COLS = ["ic", "accuracy", "mae", "rmse", "mape"]
_METRIC_HEADERS = ["IC", "Accuracy", "MAE", "RMSE", "MAPE"]


def write_latex_table(rows: list[dict], out_path: str) -> None:
    col_fmt = "llrrrrr"
    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{Ablation study results}",
        r"  \label{tab:ablation}",
        rf"  \begin{{tabular}}{{{col_fmt}}}",
        r"    \toprule",
        r"    \textbf{ID} & \textbf{Name} & \textbf{IC} & \textbf{Accuracy} "
        r"& \textbf{MAE} & \textbf{RMSE} & \textbf{MAPE} \\",
        r"    \midrule",
    ]

    for r in rows:
        cells = [r["experiment"], r["name"].replace("&", r"\&")]
        for m in _METRIC_COLS:
            v = r[m]
            cells.append(f"{v:.4f}" if not np.isnan(v) else "--")
        lines.append("    " + " & ".join(cells) + r" \\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Statistical significance tests
# ---------------------------------------------------------------------------

def _find_prediction_csvs(experiment: str, name: str) -> tuple[str | None, str | None]:
    """Return (pred_path, label_path) or (None, None) if not found."""
    # Build a glob pattern from experiment ID and name
    safe_name = name.replace(" ", "_").replace("+", "_").replace("/", "_")
    pattern_pred = os.path.join(
        REPO_ROOT,
        "output",
        f"ablation_{experiment}_*",
        "regression",
        "regression_pred_last_step.csv",
    )
    matches = glob.glob(pattern_pred)
    if not matches:
        # Try without ablation_ prefix
        pattern_pred2 = os.path.join(
            REPO_ROOT,
            "output",
            f"*{experiment}*",
            "regression",
            "regression_pred_last_step.csv",
        )
        matches = glob.glob(pattern_pred2)

    if not matches:
        return None, None

    pred_path = matches[0]
    label_path = pred_path.replace("regression_pred_last_step.csv",
                                   "regression_label_last_step.csv")
    if not os.path.exists(label_path):
        return None, None

    return pred_path, label_path


def _load_regression_csv(path: str) -> dict[str, np.ndarray]:
    """Load a regression pred/label CSV.

    Expected columns: date, instrument, value   (or similar).
    Returns dict keyed by date with array of values.
    """
    dates: dict[str, list[float]] = {}
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        # Detect column positions heuristically
        h_lower = [h.lower().strip() for h in header]
        date_col = next(
            (i for i, h in enumerate(h_lower) if "date" in h or "time" in h), 0
        )
        val_col = next(
            (i for i, h in enumerate(h_lower)
             if h in ("score", "label", "pred", "value", "prediction")),
            len(h_lower) - 1,
        )
        for row in reader:
            if not row:
                continue
            try:
                d = row[date_col].strip()
                v = float(row[val_col])
                dates.setdefault(d, []).append(v)
            except (ValueError, IndexError):
                continue
    return {d: np.array(v) for d, v in dates.items()}


def run_significance_tests(rows: list[dict]) -> None:
    print("\n=== Statistical Significance Tests (Spearman IC, H0: IC = 0) ===\n")
    for r in rows:
        pred_path, label_path = _find_prediction_csvs(r["experiment"], r["name"])
        if pred_path is None:
            print(f"  {r['experiment']} ({r['name']}): prediction CSVs not found — skipping")
            continue

        preds_by_date = _load_regression_csv(pred_path)
        labels_by_date = _load_regression_csv(label_path)

        common_dates = sorted(set(preds_by_date) & set(labels_by_date))
        if len(common_dates) < 5:
            print(f"  {r['experiment']}: fewer than 5 common dates — skipping")
            continue

        daily_ics = []
        for d in common_dates:
            p = preds_by_date[d]
            l = labels_by_date[d]
            # Align lengths
            n = min(len(p), len(l))
            if n < 2:
                continue
            rho, _ = stats.spearmanr(p[:n], l[:n])
            if not np.isnan(rho):
                daily_ics.append(rho)

        if len(daily_ics) < 3:
            print(f"  {r['experiment']}: insufficient valid days — skipping")
            continue

        ic_arr = np.array(daily_ics)
        t_stat, p_val = stats.ttest_1samp(ic_arr, popmean=0.0)
        mean_ic = ic_arr.mean()
        std_ic = ic_arr.std(ddof=1)
        icir = mean_ic / std_ic if std_ic > 0 else float("nan")

        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else
              ("*" if p_val < 0.05 else ""))
        print(
            f"  {r['experiment']:4s}  mean_IC={mean_ic:+.4f}  std={std_ic:.4f}  "
            f"ICIR={icir:+.4f}  t={t_stat:+.3f}  p={p_val:.4f}  {sig}"
        )

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: ablation results CSV not found at {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    rows = load_ablation_results(CSV_PATH)
    print(f"Loaded {len(rows)} experiment(s) from {CSV_PATH}:")
    for r in rows:
        print(
            f"  {r['experiment']:4s}  {r['name']:<40s}  "
            f"IC={r['ic']:+.6f}" if not np.isnan(r["ic"])
            else f"  {r['experiment']:4s}  {r['name']:<40s}  IC=N/A"
        )

    print("\nGenerating figures:")

    plot_ic_comparison(
        rows,
        os.path.join(FIGURES_DIR, "ic_comparison.png"),
    )

    plot_metrics_heatmap(
        rows,
        os.path.join(FIGURES_DIR, "metrics_heatmap.png"),
    )

    plot_improvement_waterfall(
        rows,
        os.path.join(FIGURES_DIR, "improvement_waterfall.png"),
    )

    write_latex_table(
        rows,
        os.path.join(FIGURES_DIR, "ablation_table.tex"),
    )

    run_significance_tests(rows)

    print("Done.")


if __name__ == "__main__":
    main()
