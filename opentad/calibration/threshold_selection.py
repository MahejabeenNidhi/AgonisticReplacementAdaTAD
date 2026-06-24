"""
Threshold selection utilities focused on period-level count comparison
and density-aware transfer.
"""

import re
import json
import numpy as np
from collections import defaultdict


def _get_period(clip_name):
    """Extract observation period from clip name."""
    match = re.match(r"(.+)_clip\d+$", clip_name)
    return match.group(1) if match else clip_name


def _temporal_iou(seg_a, seg_b):
    inter_start = max(seg_a[0], seg_b[0])
    inter_end = min(seg_a[1], seg_b[1])
    inter = max(0.0, inter_end - inter_start)
    union = (seg_a[1] - seg_a[0]) + (seg_b[1] - seg_b[0]) - inter
    return inter / max(union, 1e-8)


def _greedy_nms(preds, iou_thresh):
    """preds: list of (abs_start, abs_end, score); returns deduplicated."""
    if not preds:
        return []
    preds_sorted = sorted(preds, key=lambda x: x[2], reverse=True)
    keep = []
    for p in preds_sorted:
        if not any(_temporal_iou(p[:2], k[:2]) >= iou_thresh for k in keep):
            keep.append(p)
    return keep


def compute_period_counts_at_threshold(
    result_dict,
    annotation_path,
    subset,
    threshold,
    nms_iou_threshold=0.3,
):
    """
    Compute per-period predicted and GT counts at a given raw score threshold.
    """
    with open(annotation_path, "r") as f:
        ann_data = json.load(f).get("database", {})

    gt_counts = defaultdict(int)
    valid_clips = set()
    for clip_name, clip_info in ann_data.items():
        if subset is not None and clip_info.get("subset", "") != subset:
            continue
        valid_clips.add(clip_name)
        period = _get_period(clip_name)
        gt_counts[period] += len(clip_info.get("annotations", []))

    # Convert to absolute time, grouped by period
    period_preds = defaultdict(list)
    for clip_name, predictions in result_dict.items():
        if clip_name not in valid_clips:
            continue
        source_start = ann_data[clip_name].get("source_start", 0.0)
        period = _get_period(clip_name)
        for pred in predictions:
            if pred["score"] >= threshold:
                abs_start = source_start + pred["segment"][0]
                abs_end = source_start + pred["segment"][1]
                period_preds[period].append((abs_start, abs_end, pred["score"]))

    all_periods = sorted(set(list(gt_counts.keys()) + list(period_preds.keys())))

    period_gt = {}
    period_pred = {}
    total_ae = 0.0
    for period in all_periods:
        gt_c = gt_counts.get(period, 0)
        preds = period_preds.get(period, [])
        deduped = _greedy_nms(preds, nms_iou_threshold)
        pred_c = len(deduped)
        period_gt[period] = gt_c
        period_pred[period] = pred_c
        total_ae += abs(pred_c - gt_c)

    mae = total_ae / max(len(all_periods), 1)

    return {
        'period_gt': period_gt,
        'period_pred': period_pred,
        'total_gt': sum(period_gt.values()),
        'total_pred': sum(period_pred.values()),
        'mae': mae,
        'periods': all_periods,
    }


def optimize_threshold_for_period_counts(
    result_dict,
    annotation_path,
    subset,
    nms_iou_threshold=0.3,
    n_candidates=1000,
):
    """
    Find the raw score threshold minimising period-level count MAE.
    """
    with open(annotation_path, "r") as f:
        ann_data = json.load(f).get("database", {})

    gt_counts = defaultdict(int)
    valid_clips = set()
    for clip_name, clip_info in ann_data.items():
        if subset is not None and clip_info.get("subset", "") != subset:
            continue
        valid_clips.add(clip_name)
        period = _get_period(clip_name)
        gt_counts[period] += len(clip_info.get("annotations", []))

    # Collect all predictions with absolute times
    period_preds = defaultdict(list)
    all_scores = []
    for clip_name, predictions in result_dict.items():
        if clip_name not in valid_clips:
            continue
        source_start = ann_data[clip_name].get("source_start", 0.0)
        period = _get_period(clip_name)
        for pred in predictions:
            abs_start = source_start + pred["segment"][0]
            abs_end = source_start + pred["segment"][1]
            period_preds[period].append((abs_start, abs_end, pred["score"]))
            all_scores.append(pred["score"])

    all_periods = sorted(set(list(gt_counts.keys()) + list(period_preds.keys())))
    all_scores = np.array(all_scores)

    if len(all_scores) == 0:
        return {'optimal_threshold': 0.5, 'optimal_mae': 0.0,
                'all_thresholds': [], 'all_maes': []}

    # Generate thresholds: percentiles + linear grid
    percentile_thresholds = np.percentile(all_scores, np.linspace(0, 100, n_candidates // 2))
    linear_thresholds = np.linspace(0.001, all_scores.max() + 0.01, n_candidates // 2)
    thresholds = np.unique(np.concatenate([percentile_thresholds, linear_thresholds]))
    thresholds = np.sort(thresholds)

    all_maes = []
    for thresh in thresholds:
        total_ae = 0.0
        for period in all_periods:
            gt_c = gt_counts.get(period, 0)
            preds = period_preds.get(period, [])
            filtered = [p for p in preds if p[2] >= thresh]
            deduped = _greedy_nms(filtered, nms_iou_threshold)
            total_ae += abs(len(deduped) - gt_c)
        mae = total_ae / max(len(all_periods), 1)
        all_maes.append(mae)

    all_maes = np.array(all_maes)
    best_idx = np.argmin(all_maes)

    return {
        'optimal_threshold': float(thresholds[best_idx]),
        'optimal_mae': float(all_maes[best_idx]),
        'all_thresholds': thresholds.tolist(),
        'all_maes': all_maes.tolist(),
    }
