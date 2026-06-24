"""
Detection Operating Curve: maps raw score thresholds to per-detection
precision, recall, and F1 on a validation set.
"""

import numpy as np
from collections import defaultdict


def _segment_iou(target, candidates):
    """Compute tIoU between one target segment and N candidate segments."""
    candidates = np.asarray(candidates, dtype=np.float64)
    if candidates.ndim == 1:
        candidates = candidates.reshape(1, 2)
    tt1 = np.maximum(target[0], candidates[:, 0])
    tt2 = np.minimum(target[1], candidates[:, 1])
    inter = np.clip(tt2 - tt1, 0, None)
    union = (
        (candidates[:, 1] - candidates[:, 0])
        + (target[1] - target[0])
        - inter
    )
    return inter / np.clip(union, 1e-8, None)


def label_detections(result_dict, annotation_path, subset, tiou_match=0.3):
    """
    Label every detection as TP or FP by greedy one-to-one matching at
    tIoU >= tiou_match. Also returns which video each detection belongs to.

    Uses greedy matching (highest-scoring detection first) per video,
    which is the standard VOC protocol.
    """
    import json

    with open(annotation_path, "r") as f:
        ann_data = json.load(f).get("database", {})

    scores_all = []
    is_tp_all = []
    video_ids_all = []
    n_gt_total = 0
    gt_per_video = {}

    for clip_name, predictions in result_dict.items():
        if clip_name not in ann_data:
            continue
        clip_info = ann_data[clip_name]
        if subset is not None and clip_info.get("subset", "") != subset:
            continue

        # GT segments in clip-local seconds
        gt_segs = np.array(
            [[float(a["segment"][0]), float(a["segment"][1])]
             for a in clip_info.get("annotations", [])],
            dtype=np.float64,
        ).reshape(-1, 2)

        n_gt = len(gt_segs)
        n_gt_total += n_gt
        gt_per_video[clip_name] = n_gt

        # Sort predictions by descending score for greedy matching
        preds_sorted = sorted(predictions, key=lambda x: x["score"], reverse=True)
        gt_matched = np.zeros(n_gt, dtype=bool)

        for pred in preds_sorted:
            score = pred["score"]
            is_tp = False

            if n_gt > 0:
                pred_seg = np.array([pred["segment"][0], pred["segment"][1]])
                tiou_arr = _segment_iou(pred_seg, gt_segs)

                # Find best unmatched GT
                sorted_gt_idx = np.argsort(-tiou_arr)
                for gt_idx in sorted_gt_idx:
                    if tiou_arr[gt_idx] < tiou_match:
                        break
                    if not gt_matched[gt_idx]:
                        is_tp = True
                        gt_matched[gt_idx] = True
                        break

            scores_all.append(score)
            is_tp_all.append(is_tp)
            video_ids_all.append(clip_name)

    return (
        np.array(scores_all, dtype=np.float64),
        np.array(is_tp_all, dtype=bool),
        video_ids_all,
        n_gt_total,
        gt_per_video,
    )


def compute_detection_operating_curve(
    scores,
    is_tp,
    n_gt_total,
    n_thresholds=2000,
):
    """
    Compute precision, recall, and F-beta at various score thresholds.

    This is the core of the operating curve: for each threshold, count
    how many detections are TP vs FP, and how many GT events are recalled.
    """
    if len(scores) == 0:
        empty = np.array([])
        return {
            'thresholds': empty, 'precision': empty, 'recall': empty,
            'f1': empty, 'f2': empty, 'n_accepted': empty, 'n_tp_above': empty,
        }

    # Use unique score values + a linear grid for comprehensive coverage
    unique_scores = np.unique(scores)
    linear_grid = np.linspace(scores.min(), scores.max(), n_thresholds)
    thresholds = np.unique(np.concatenate([unique_scores, linear_grid]))
    thresholds = np.sort(thresholds)

    precisions = []
    recalls = []
    f1_scores = []
    f2_scores = []
    n_accepted_list = []
    n_tp_list = []

    for thresh in thresholds:
        mask = scores >= thresh
        n_above = mask.sum()
        n_tp_above = is_tp[mask].sum() if n_above > 0 else 0

        if n_above == 0:
            prec = 1.0  # No predictions = no FP
            rec = 0.0
        else:
            prec = n_tp_above / n_above
            rec = n_tp_above / max(n_gt_total, 1)

        # F-beta scores
        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
            f2 = 5 * prec * rec / (4 * prec + rec)  # F2 weighs recall 2x
        else:
            f1 = 0.0
            f2 = 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
        f2_scores.append(f2)
        n_accepted_list.append(n_above)
        n_tp_list.append(n_tp_above)

    return {
        'thresholds': thresholds,
        'precision': np.array(precisions),
        'recall': np.array(recalls),
        'f1': np.array(f1_scores),
        'f2': np.array(f2_scores),
        'n_accepted': np.array(n_accepted_list),
        'n_tp_above': np.array(n_tp_list),
    }


def select_threshold_by_target_recall(curve, target_recall, min_precision=0.0):
    """
    Select the HIGHEST threshold that achieves at least `target_recall`.
    Higher threshold = fewer false alarms = more practical.

    Optionally require minimum precision.
    """
    thresholds = curve['thresholds']
    recalls = curve['recall']
    precisions = curve['precision']

    # Find indices where recall >= target and precision >= min
    valid_mask = (recalls >= target_recall) & (precisions >= min_precision)

    if not valid_mask.any():
        # Target not achievable; return the threshold giving maximum recall
        best_idx = np.argmax(recalls)
        return {
            'threshold': float(thresholds[best_idx]),
            'recall': float(recalls[best_idx]),
            'precision': float(precisions[best_idx]),
            'f1': float(curve['f1'][best_idx]),
            'f2': float(curve['f2'][best_idx]),
            'n_accepted': int(curve['n_accepted'][best_idx]),
            'achieved': False,
        }

    # Among valid indices, pick the highest threshold (most conservative)
    valid_indices = np.where(valid_mask)[0]
    best_idx = valid_indices[np.argmax(thresholds[valid_indices])]

    return {
        'threshold': float(thresholds[best_idx]),
        'recall': float(recalls[best_idx]),
        'precision': float(precisions[best_idx]),
        'f1': float(curve['f1'][best_idx]),
        'f2': float(curve['f2'][best_idx]),
        'n_accepted': int(curve['n_accepted'][best_idx]),
        'achieved': True,
    }


def select_threshold_by_max_f2(curve, min_recall=0.0):
    """
    Select threshold that maximises F2 score (recall-weighted F-measure).
    F2 weighs recall twice as much as precision — appropriate for
    ethological screening where missing events is worse than false alarms.
    """
    f2 = curve['f2']
    recalls = curve['recall']

    valid_mask = recalls >= min_recall
    if not valid_mask.any():
        best_idx = np.argmax(f2)
    else:
        masked_f2 = np.where(valid_mask, f2, -1)
        best_idx = np.argmax(masked_f2)

    return {
        'threshold': float(curve['thresholds'][best_idx]),
        'recall': float(curve['recall'][best_idx]),
        'precision': float(curve['precision'][best_idx]),
        'f1': float(curve['f1'][best_idx]),
        'f2': float(f2[best_idx]),
        'n_accepted': int(curve['n_accepted'][best_idx]),
    }


def apply_isotonic_calibration(scores, is_tp, new_scores=None):
    """
    Fit isotonic regression: maps raw score → P(TP | score).
    This is for interpretability only — the threshold selection
    works directly on raw scores.
    """
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(scores, is_tp.astype(float))

    calibrator = iso.predict

    if new_scores is None:
        calibrated = calibrator(scores)
    else:
        calibrated = calibrator(np.asarray(new_scores))

    return calibrated, calibrator
