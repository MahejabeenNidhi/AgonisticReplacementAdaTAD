# opentad/evaluations/period_counts.py

import os
import re
import json
import numpy as np
from collections import defaultdict


def compute_period_event_counts(
    result_dict,
    annotation_path,
    score_thresholds,
    subset,
    nms_iou_threshold=0.3,
    logger=None,
    work_dir=None,
    named_thresholds=None,
):
    """
    Aggregate per-clip predictions into observation-period event counts
    after converting to absolute time and deduplicating across overlapping
    clips via greedy NMS.  Compares predicted counts against ground-truth
    annotation counts and reports MAE per score threshold.

    Parameters
    ----------
    result_dict : dict
        {clip_name: [{"segment": [start_sec, end_sec], "score": float,
                      "label": str}, ...]}
        Predictions in clip-local seconds (0–300).
    annotation_path : str
        Path to annotation JSON (contains source_start and annotations
        per clip).
    score_thresholds : list of float
        Score thresholds at which to count events.
    subset : str or None
        Subset to consider ("testing", "validation", etc.).
    nms_iou_threshold : float
        tIoU threshold for greedy cross-clip deduplication NMS.
    logger : logging.Logger or None
    work_dir : str or None
        If provided, saves a text table to {work_dir}/period_event_counts.txt.
    """

    pprint = logger.info if logger else print

    # ------------------------------------------------------------------
    # 1. Load annotation JSON
    # ------------------------------------------------------------------
    with open(annotation_path, "r") as f:
        ann_data = json.load(f).get("database", {})

    # ------------------------------------------------------------------
    # 2. Helper: extract observation period from clip name
    #    YYYYMMDD_HHMM-HHMM_CamX_clipNNNN  ->  YYYYMMDD_HHMM-HHMM_CamX
    # ------------------------------------------------------------------
    def get_period(clip_name):
        match = re.match(r"(.+)_clip\d+$", clip_name)
        return match.group(1) if match else clip_name

    # ------------------------------------------------------------------
    # 3. Build GT counts per period (sum annotations across member clips)
    # ------------------------------------------------------------------
    gt_counts = defaultdict(int)
    valid_clips = set()

    for clip_name, clip_info in ann_data.items():
        if subset is not None and clip_info.get("subset", "") != subset:
            continue
        valid_clips.add(clip_name)
        period = get_period(clip_name)
        gt_counts[period] += len(clip_info.get("annotations", []))

    # ------------------------------------------------------------------
    # 4. Convert predictions to absolute time, grouped by period
    # ------------------------------------------------------------------
    period_abs_preds = defaultdict(list)  # {period: [(abs_start, abs_end, score), ...]}

    for clip_name, predictions in result_dict.items():
        if clip_name not in valid_clips:
            continue
        source_start = ann_data[clip_name].get("source_start", 0)
        period = get_period(clip_name)
        for pred in predictions:
            abs_start = source_start + pred["segment"][0]
            abs_end = source_start + pred["segment"][1]
            period_abs_preds[period].append((abs_start, abs_end, pred["score"]))

    # ------------------------------------------------------------------
    # 5a. Greedy NMS (self-contained, no external dependency)
    # ------------------------------------------------------------------
    def _iou(a, b):
        inter_s = max(a[0], b[0])
        inter_e = min(a[1], b[1])
        inter = max(0.0, inter_e - inter_s)
        union = (a[1] - a[0]) + (b[1] - b[0]) - inter
        return inter / max(union, 1e-8)

    def _greedy_nms(preds, iou_thresh):
        """preds: list of (start, end, score); returns deduplicated subset."""
        keep = []
        for p in sorted(preds, key=lambda x: x[2], reverse=True):
            if not any(_iou(p[:2], k[:2]) >= iou_thresh for k in keep):
                keep.append(p)
        return keep

    # ------------------------------------------------------------------
    # 5b. Build unified (value, label) list from numeric + named thresholds
    #     Named thresholds override the numeric label if values collide.
    # ------------------------------------------------------------------
    _thresh_map = {t: f"{t:.2f}" for t in score_thresholds}
    if named_thresholds:
        for val, name in named_thresholds:
            _thresh_map[val] = name  # replaces numeric label on exact collision

    all_thresh_pairs = sorted(_thresh_map.items(), key=lambda x: x[0])
    thresh_values = [v for v, _ in all_thresh_pairs]  # floats, for computation
    thresh_labels = [lbl for _, lbl in all_thresh_pairs]  # strings, for display

    # ------------------------------------------------------------------
    # 6. Count predictions per period at each threshold
    # ------------------------------------------------------------------
    all_periods = sorted(
        set(list(gt_counts.keys()) + list(period_abs_preds.keys()))
    )

    pred_counts = {t: {} for t in thresh_values}
    abs_errors = {t: [] for t in thresh_values}
    total_pred = {t: 0 for t in thresh_values}
    total_gt = 0

    for period in all_periods:
        gt_c = gt_counts.get(period, 0)
        total_gt += gt_c
        preds = period_abs_preds.get(period, [])
        for thresh in thresh_values:
            filtered = [p for p in preds if p[2] >= thresh]
            deduped = _greedy_nms(filtered, nms_iou_threshold)
            count = len(deduped)
            pred_counts[thresh][period] = count
            total_pred[thresh] += count
            abs_errors[thresh].append(abs(count - gt_c))

    # ------------------------------------------------------------------
    # 7. Build formatted table
    # ------------------------------------------------------------------
    name_w = max(38, max((len(v) for v in all_periods), default=38) + 2)
    thresh_headers = [f"Pred@{lbl}" for lbl in thresh_labels]
    col_w = max(9, max(len(h) for h in thresh_headers))  # dynamic width

    header = (
            f"{'Period':<{name_w}} | {'GT':>4} | "
            + " | ".join(f"{h:>{col_w}}" for h in thresh_headers)
    )
    separator = "-" * len(header)
    lines = [
        "",
        "=" * len(header),
        (
            "Period Event Counts — GT vs Predicted "
            f"(NMS tIoU >= {nms_iou_threshold:.2f})"
        ),
        "=" * len(header),
        header,
        separator,
    ]
    for period in all_periods:
        gt_c = gt_counts.get(period, 0)
        row = f"{period:<{name_w}} | {gt_c:>4}"
        for thresh in thresh_values:
            row += f" | {pred_counts[thresh].get(period, 0):>{col_w}}"
        lines.append(row)
    lines.append(separator)
    total_row = f"{'TOTAL':<{name_w}} | {total_gt:>4}"
    for thresh in thresh_values:
        total_row += f" | {total_pred[thresh]:>{col_w}}"
    lines.append(total_row)
    mae_row = f"{'MAE (vs GT)':<{name_w}} | {'':>4}"
    for thresh in thresh_values:
        mae_val = float(np.mean(abs_errors[thresh])) if abs_errors[thresh] else 0.0
        mae_row += f" | {mae_val:>{col_w}.2f}"
    lines.append(mae_row)
    lines.append("=" * len(header))

    msg = "\n".join(lines)
    pprint(msg)

    # ------------------------------------------------------------------
    # 8. Optionally persist to disk
    # ------------------------------------------------------------------
    if work_dir is not None:
        out_path = os.path.join(work_dir, "period_event_counts.txt")
        with open(out_path, "w") as f:
            f.write(msg + "\n")
        pprint(f"Period event counts saved to: {out_path}")

    return dict(gt_counts), pred_counts