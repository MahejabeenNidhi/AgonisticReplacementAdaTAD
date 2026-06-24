#!/usr/bin/env python3
"""
Analyse temporal action detection counts from the .txt output file.

Usage
-----
    # All thresholds (default)
    python analyse_counts.py counts.txt

    # Specific thresholds
    python analyse_counts.py counts.txt -t GT Pred@0.30 Pred@0.50

    # List available thresholds and exit
    python analyse_counts.py counts.txt --list-thresholds

Outputs
-------
  • Console : descriptive statistics + paired t-test table for every threshold
  • File    : 'agonistic_replacement_boxplots.png'

Requirements
------------
    pip install numpy scipy matplotlib
"""

from __future__ import annotations

import argparse
import re
import sys
from math import ceil

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ── Configuration ─────────────────────────────────────────────────────────────
OBS_HOURS: float = 2.0   # each observation window is 2 hours → rate = count / 2

# (pre-treatment period, post-treatment period)
PAIRS: list[tuple[str, str]] = [
    ("20240915_1730-1930_Cam11", "20240916_1740-1940_Cam11"),
    ("20240915_1820-2020_Cam12", "20240916_1845-2045_Cam12"),
    ("20241027_1800-2000_Cam11", "20241028_1820-2020_Cam11"),
    ("20241028_1830-2030_Cam12", "20241103_1815-2015_Cam12"),
    ("20241103_1810-2010_Cam11", "20241104_1810-2010_Cam11"),
    ("20241103_1815-2015_Cam12", "20241104_1815-2015_Cam12"),
    ("20251105_1700-1900_Cam11", "20251106_1700-1900_Cam11"),
]

# Number of cows in each group replicate (same order as PAIRS).
# Pre- and post-treatment periods within a pair use the same group,
# so one value per pair suffices.
# Replace these with your actual group sizes.
GROUP_SIZES: list[int] = [
    12,  # Pair 1: 20240915 Cam11
    11,  # Pair 2: 20240915 Cam12
    12,  # Pair 3: 20241027 Cam11
    10,  # Pair 4: 20241028 Cam12
    12,  # Pair 5: 20241103 Cam11
    9,   # Pair 6: 20241103 Cam12
    11,  # Pair 7: 20251105 Cam11
]
# ─────────────────────────────────────────────────────────────────────────────


# ── 1. Argument parsing ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Box plots + paired t-tests for agonistic replacement counts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python analyse_counts.py counts.txt\n"
            "  python analyse_counts.py counts.txt -t GT Pred@0.30 Pred@0.50\n"
            "  python analyse_counts.py counts.txt --list-thresholds\n"
        ),
    )
    parser.add_argument(
        "filepath",
        nargs="?",
        default="counts.txt",
        help="Path to the counts .txt file (default: counts.txt)",
    )
    parser.add_argument(
        "-t", "--thresholds",
        nargs="+",
        metavar="THRESHOLD",
        default=None,
        help=(
            "One or more threshold column names to plot, e.g. "
            "'-t GT Pred@0.30 Pred@0.50'. "
            "If omitted, all available thresholds are used."
        ),
    )
    parser.add_argument(
        "--list-thresholds",
        action="store_true",
        help="Print available threshold column names and exit.",
    )
    parser.add_argument(
        "-o", "--output",
        default="agonistic_replacement_boxplots.png",
        metavar="FILE",
        help="Output image filename (default: agonistic_replacement_boxplots.png)",
    )
    parser.add_argument(
        "--no-per-cow",
        action="store_true",
        help=(
            "If set, compute rates as count / hours (group-level) rather than "
            "count / hours / n_cows (per-cow). Default is per-cow to match "
            "the ethologist's normalisation in Section 3.1."
        ),
    )
    return parser.parse_args()


# ── 2. File parsing ───────────────────────────────────────────────────────────

def parse_counts_file(filepath: str) -> tuple[dict, list[str]]:
    """
    Read the period-event-counts .txt file.

    Returns
    -------
    data      : {period_name: {threshold_label: count (int)}}
    col_names : ordered list of threshold column names  (e.g. ['GT', 'Pred@0.10', …])
    """
    with open(filepath, encoding="utf-8") as fh:
        lines = fh.readlines()

    # Locate the header row (first line that matches "Period |")
    header_idx = next(
        (i for i, ln in enumerate(lines) if re.match(r"\s*Period\s*\|", ln)),
        None,
    )
    if header_idx is None:
        raise ValueError("Cannot find 'Period |' header row in the file.")

    col_names: list[str] = [c.strip() for c in lines[header_idx].split("|")][1:]

    # Parse data rows — period names look like  YYYYMMDD_HHMM-HHMM_CamNN
    period_re = re.compile(r"^\s*(\d{8}_\d{4}-\d{4}_Cam\d+)")
    data: dict = {}

    for line in lines[header_idx + 1 :]:
        m = period_re.match(line)
        if not m:
            continue
        parts = [p.strip() for p in line.split("|")]
        period = m.group(1)
        data[period] = {}
        for j, col in enumerate(col_names, start=1):
            try:
                data[period][col] = int(parts[j])
            except (IndexError, ValueError):
                data[period][col] = None

    return data, col_names


# ── 3. Threshold selection ────────────────────────────────────────────────────

def resolve_thresholds(
    requested: list[str] | None,
    available: list[str],
) -> list[str]:
    """
    Validate and return the threshold columns to use.

    Parameters
    ----------
    requested : user-supplied list, or None (meaning "all")
    available : columns found in the file

    Raises ValueError for any unrecognised threshold name.
    """
    if requested is None:
        return available

    bad = [t for t in requested if t not in available]
    if bad:
        raise ValueError(
            f"Unknown threshold(s): {bad}\n"
            f"Available thresholds: {available}"
        )
    return requested


# ── 4. Rate arrays ────────────────────────────────────────────────────────────

def build_rate_arrays(
    data: dict,
    col_names: list[str],
    pairs: list[tuple[str, str]],
    obs_hours: float,
    group_sizes: list[int],
    per_cow: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Convert raw counts to rates for the given threshold columns.

    Parameters
    ----------
    per_cow : bool
        If True (default), rate = count / obs_hours / n_cows.
        This matches the ethologist's normalisation in Section 3.1.
        If False, rate = count / obs_hours (group-level rate).

    Returns {col: (pre_rates, post_rates)}.
    A pair is skipped silently if either period is missing from the file.
    """
    result: dict = {}
    for col in col_names:
        pre_list, post_list = [], []
        for (pre_period, post_period), n_cows in zip(pairs, group_sizes):
            pc = data.get(pre_period, {}).get(col)
            qc = data.get(post_period, {}).get(col)
            if pc is not None and qc is not None:
                denominator = obs_hours * n_cows if per_cow else obs_hours
                pre_list.append(pc / denominator)
                post_list.append(qc / denominator)
        result[col] = (
            np.array(pre_list,  dtype=float),
            np.array(post_list, dtype=float),
        )
    return result


# ── 5. Console output ─────────────────────────────────────────────────────────

def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "*** (p < 0.001)"
    if p < 0.01:
        return "**  (p < 0.01)"
    if p < 0.05:
        return "*   (p < 0.05)"
    return "n.s."


def print_summary(
    rates: dict[str, tuple[np.ndarray, np.ndarray]],
    per_cow: bool,
) -> None:
    """Print descriptive statistics and paired t-test results to stdout."""
    W = 108
    sep = "─" * W

    unit = "events / cow / hour" if per_cow else "events / hour"

    # ── Descriptive stats ──
    print(f"\n{sep}")
    print(f"DESCRIPTIVE STATISTICS  ({unit})")
    print(sep)
    print(
        f"{'Threshold':<14} │ "
        f"{'Pre mean':>9}  {'Pre range':>22}  {'Pre SD':>7} │ "
        f"{'Post mean':>10}  {'Post range':>22}  {'Post SD':>7}"
    )
    print(sep)
    for col, (pre, post) in rates.items():
        pre_rng  = f"{pre.min():.2f} – {pre.max():.2f}"
        post_rng = f"{post.min():.2f} – {post.max():.2f}"
        print(
            f"{col:<14} │ "
            f"{pre.mean():>9.2f}  {pre_rng:>22}  {pre.std(ddof=1):>7.2f} │ "
            f"{post.mean():>10.2f}  {post_rng:>22}  {post.std(ddof=1):>7.2f}"
        )

    # ── Paired t-tests ──
    print(f"\n{sep}")
    print("PAIRED t-TEST RESULTS  (pre vs post, two-tailed)")
    print(sep)
    print(
        f"{'Threshold':<14} │ {'t-statistic':>12} │ {'df':>3} │ "
        f"{'p-value':>10} │ Significance"
    )
    print(sep)
    for col, (pre, post) in rates.items():
        t_stat, p_val = stats.ttest_rel(pre, post)
        df = len(pre) - 1
        print(
            f"{col:<14} │ {t_stat:>12.4f} │ {df:>3} │ "
            f"{p_val:>10.4f} │ {_sig_stars(p_val)}"
        )
    print(sep)
    print(
        f"Note: Rates = raw counts ÷ observation window duration "
        f"({OBS_HOURS:.0f} h)"
        + (f" ÷ group size (n cows)." if per_cow else ".")
    )
    print(
        "      Stars for means; boxes = Q1 / median / Q3; "
        "whiskers = 1.5 × IQR.\n"
    )


# ── 6. Box-plot figure ────────────────────────────────────────────────────────

def _sig_bracket_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def draw_boxplots(
    rates: dict[str, tuple[np.ndarray, np.ndarray]],
    per_cow: bool,
    output_path: str = "agonistic_replacement_boxplots.png",
) -> None:
    """One subplot per threshold, each showing pre vs post box plots."""
    n       = len(rates)
    n_cols  = min(n, 4)
    n_rows  = ceil(n / n_cols)

    PRE_COLOR  = "#6BAED6"   # muted blue
    POST_COLOR = "#FC8D59"   # muted orange

    y_label = "Events / cow / hour" if per_cow else "Events / hour"

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.4 * n_cols, 5.8 * n_rows),
        squeeze=False,
    )
    flat_axes = axes.flatten()

    for idx, (col, (pre, post)) in enumerate(rates.items()):
        ax = flat_axes[idx]

        bp = ax.boxplot(
            [pre, post],
            labels=["Pre-treatment", "Post-treatment"],
            patch_artist=True,
            medianprops=dict(color="black",  linewidth=2.0),
            whiskerprops=dict(linewidth=1.4),
            capprops=dict(linewidth=1.4),
            flierprops=dict(
                marker="o", markerfacecolor="grey",
                markersize=5, linestyle="none", markeredgewidth=0.5,
            ),
            widths=0.52,
        )
        bp["boxes"][0].set_facecolor(PRE_COLOR);  bp["boxes"][0].set_alpha(0.85)
        bp["boxes"][1].set_facecolor(POST_COLOR); bp["boxes"][1].set_alpha(0.85)

        # Mean markers (★ as in the ethologist figure)
        ax.plot(1, pre.mean(),  marker="*", color="navy",        markersize=13,
                zorder=6, clip_on=False, label="Mean")
        ax.plot(2, post.mean(), marker="*", color="saddlebrown", markersize=13,
                zorder=6, clip_on=False)

        # Significance bracket above the tallest whisker / outlier
        t_stat, p_val = stats.ttest_rel(pre, post)
        df_val = len(pre) - 1

        all_vals = np.concatenate([pre, post])
        y_top  = all_vals.max()
        y_bot  = max(all_vals.min(), 0.0)
        span   = max(y_top - y_bot, 1.0)

        bk_y  = y_top + span * 0.18
        txt_y = bk_y  + span * 0.04

        ax.annotate(
            "", xy=(2, bk_y), xytext=(1, bk_y),
            arrowprops=dict(arrowstyle="-", lw=1.5, color="black"),
        )
        ax.text(
            1.5, txt_y, _sig_bracket_label(p_val),
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

        ax.set_ylim(bottom=0, top=txt_y + span * 0.18)
        ax.set_title(
            f"{col}\n"
            r"$t_{" + str(df_val) + r"}$"
            f" = {t_stat:.2f},  p = {p_val:.3f}",
            fontsize=9.5,
        )
        ax.set_ylabel(y_label if idx % n_cols == 0 else "")
        ax.yaxis.grid(True, linestyle="--", alpha=0.45)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8.5)

    # Hide any unused subplot panels
    for j in range(idx + 1, len(flat_axes)):
        flat_axes[j].set_visible(False)

    # Shared legend
    legend_handles = [
        mpatches.Patch(facecolor=PRE_COLOR,  edgecolor="black",
                       alpha=0.85, label="Pre-treatment"),
        mpatches.Patch(facecolor=POST_COLOR, edgecolor="black",
                       alpha=0.85, label="Post-treatment"),
        plt.Line2D([], [], marker="*", linestyle="none",
                   color="navy", markersize=11, label="Mean"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, 0.0),
        fontsize=10, frameon=True,
    )

    plotted = ", ".join(rates.keys())
    norm_note = "per-cow hourly rate" if per_cow else "group hourly rate"
    fig.suptitle(
        f"Agonistic Replacement Rates: Pre- vs Post-treatment ({norm_note})\n"
        f"Thresholds: {plotted}  "
        f"(n = {len(PAIRS)} pairs,  {OBS_HOURS:.0f}-hour observation windows)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.07, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved → '{output_path}'")
    plt.show()


# ── 7. Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    per_cow = not args.no_per_cow

    print(f"Parsing : {args.filepath}")
    print(f"Normalisation: {'per-cow hourly rate (count / hours / n_cows)' if per_cow else 'group hourly rate (count / hours)'}")
    data, all_col_names = parse_counts_file(args.filepath)
    print(f"  Periods detected  : {len(data)}")
    print(f"  Threshold columns : {all_col_names}")

    # --list-thresholds: just print and exit
    if args.list_thresholds:
        print("\nAvailable thresholds:")
        for col in all_col_names:
            print(f"  {col}")
        sys.exit(0)

    # Validate / resolve the requested subset
    try:
        selected_cols = resolve_thresholds(args.thresholds, all_col_names)
    except ValueError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)

    # Validate group sizes
    if per_cow and len(GROUP_SIZES) != len(PAIRS):
        print(
            f"\nError: GROUP_SIZES has {len(GROUP_SIZES)} entries but "
            f"PAIRS has {len(PAIRS)} entries. They must match.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  Plotting          : {selected_cols}")
    if per_cow:
        print(f"  Group sizes       : {GROUP_SIZES}")

    rates = build_rate_arrays(
        data, selected_cols, PAIRS, OBS_HOURS, GROUP_SIZES, per_cow=per_cow
    )
    print_summary(rates, per_cow=per_cow)
    draw_boxplots(rates, per_cow=per_cow, output_path=args.output)


if __name__ == "__main__":
    main()
