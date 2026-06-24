#!/usr/bin/env python3
"""
Standalone Review Time Analysis with Built-in Inference
========================================================
Runs model inference (like test.py) then computes how much video an
ethologist would need to review at a given recall operating point.

Padding is estimated from validation set boundary errors (methodologically
correct), then applied to the test set review time computation.

Usage:
    torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:0 \
        tools/analyze_review_time.py \
        configs/adatad/displacement/e2e_displacement_2min_videomaev2_b_160x2_160_adapter_predecoded.py \
        --checkpoint exps/displacement/adatad/.../checkpoint/best.pth \
        --subset testing \
        --padding_subset validation \
        --target_recall 0.8 \
        --tiou_match 0.3 \
        --padding_percentile 90
"""

import os
import sys
import json
import copy
import argparse
import numpy as np
from collections import defaultdict

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import torch
import torch.distributed as dist
import tqdm
from torch.nn.parallel import DistributedDataParallel
from mmengine.config import Config, DictAction

from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader
from opentad.datasets.base import SlidingWindowDataset
from opentad.models.utils.post_processing import build_classifier, batched_nms
from opentad.utils import update_workdir, set_seed, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference + review time analysis (standalone)"
    )
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--checkpoint", type=str, default="none", help="checkpoint path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--subset", type=str, default="testing",
                        help="Subset to analyze for review time (default: testing)")
    parser.add_argument("--padding_subset", type=str, default="validation",
                        help="Subset to estimate padding from (default: validation)")
    parser.add_argument("--target_recall", type=float, default=0.8,
                        help="Target event detection recall (default: 0.8)")
    parser.add_argument("--tiou_match", type=float, default=0.3,
                        help="tIoU threshold for TP matching (default: 0.3)")
    parser.add_argument("--nms_iou", type=float, default=0.3,
                        help="NMS IoU for cross-clip dedup (default: 0.3)")
    parser.add_argument("--padding_percentile", type=float, default=90,
                        help="Percentile of boundary errors for padding (default: 90)")
    parser.add_argument("--fixed_padding", type=float, default=None,
                        help="Override: use fixed padding in seconds (skip validation inference)")
    parser.add_argument("--padding_candidates", type=float, nargs="+",
                        default=[5, 10, 15, 20, 25, 30],
                        help="Padding values to compare")
    parser.add_argument("--threshold_override", type=float, default=None,
                        help="Override: use this score threshold directly")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save result_dict to JSON for reuse")
    parser.add_argument("--load_predictions", type=str, default=None,
                        help="Skip test inference; load predictions from this JSON file")
    parser.add_argument("--load_val_predictions", type=str, default=None,
                        help="Skip val inference; load val predictions from this JSON file")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction,
                        help="override config settings")
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════
# Inference (embedded from test.py logic)
# ══════════════════════════════════════════════════════════════════════

def gather_ddp_results(world_size, result_dict, post_cfg):
    """Gather results from all DDP ranks and optionally apply sliding window NMS."""
    gather_dict_list = [None for _ in range(world_size)]
    dist.all_gather_object(gather_dict_list, result_dict)

    result_dict = {}
    for i in range(world_size):
        for k, v in gather_dict_list[i].items():
            if k in result_dict:
                result_dict[k].extend(v)
            else:
                result_dict[k] = v

    # Sliding window NMS if needed
    if post_cfg.sliding_window and post_cfg.nms is not None:
        tmp_result_dict = {}
        for k, v in result_dict.items():
            segments = torch.Tensor([data["segment"] for data in v])
            scores = torch.Tensor([data["score"] for data in v])
            labels = []
            class_idx = []
            for data in v:
                if data["label"] not in class_idx:
                    class_idx.append(data["label"])
                labels.append(class_idx.index(data["label"]))
            labels = torch.Tensor(labels)
            segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)
            results_per_video = []
            for segment, label, score in zip(segments, labels, scores):
                results_per_video.append(
                    dict(
                        segment=[round(seg.item(), 2) for seg in segment],
                        label=class_idx[int(label.item())],
                        score=round(score.item(), 4),
                    )
                )
            tmp_result_dict[k] = results_per_video
        result_dict = tmp_result_dict

    return result_dict


def build_model_and_load(cfg, args, logger):
    """Build model and load checkpoint. Returns model on GPU with DDP."""
    model = build_detector(cfg.model)
    model = model.to(args.local_rank)
    model = DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank
    )

    # Load checkpoint
    if args.checkpoint != "none":
        checkpoint_path = args.checkpoint
    elif "test_epoch" in cfg.inference.keys():
        checkpoint_path = os.path.join(
            cfg.work_dir, f"checkpoint/epoch_{cfg.inference.test_epoch}.pth"
        )
    else:
        checkpoint_path = os.path.join(cfg.work_dir, "checkpoint/best.pth")

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    device = f"cuda:{args.local_rank}"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")

    use_ema = getattr(cfg.solver, "ema", False)
    if use_ema:
        model.load_state_dict(checkpoint["state_dict_ema"])
        logger.info("Using Model EMA weights")
    else:
        model.load_state_dict(checkpoint["state_dict"])

    return model


def run_inference_on_subset(cfg, args, model, subset_name, logger):
    """Run model inference on a specific subset and return result_dict."""
    # Build dataset for the requested subset
    test_cfg = copy.deepcopy(cfg.dataset.test)
    test_cfg.subset_name = subset_name
    test_dataset = build_dataset(test_cfg, default_args=dict(logger=logger))
    test_loader = build_dataloader(
        test_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=False,
        drop_last=False,
        seed=args.seed,
        **cfg.solver.test,
    )

    use_amp = getattr(cfg.solver, "amp", False)

    # Setup post-processing
    cfg.inference["folder"] = os.path.join(cfg.work_dir, "outputs")
    external_cls = None
    if "external_cls" in cfg.post_processing:
        ext_cls_cfg = cfg.post_processing.external_cls
        if ext_cls_cfg is not None:
            if isinstance(ext_cls_cfg, (list, tuple)):
                external_cls = list(ext_cls_cfg)
            else:
                external_cls = build_classifier(ext_cls_cfg)

    cfg.post_processing.sliding_window = isinstance(test_loader.dataset, SlidingWindowDataset)

    # Run inference
    model.eval()
    result_dict = {}
    desc = f"Inference [{subset_name}]"
    for data_dict in tqdm.tqdm(test_loader, disable=(args.rank != 0), desc=desc):
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                results = model(
                    **data_dict,
                    return_loss=False,
                    infer_cfg=cfg.inference,
                    post_cfg=cfg.post_processing,
                    ext_cls=external_cls,
                )
        for k, v in results.items():
            if k in result_dict:
                result_dict[k].extend(v)
            else:
                result_dict[k] = v

    result_dict = gather_ddp_results(args.world_size, result_dict, cfg.post_processing)
    return result_dict


# ══════════════════════════════════════════════════════════════════════
# Analysis utilities
# ══════════════════════════════════════════════════════════════════════

def segment_iou(target, candidates):
    """Compute tIoU between one segment and N candidates."""
    candidates = np.asarray(candidates, dtype=np.float64).reshape(-1, 2)
    tt1 = np.maximum(target[0], candidates[:, 0])
    tt2 = np.minimum(target[1], candidates[:, 1])
    inter = np.clip(tt2 - tt1, 0, None)
    union = (candidates[:, 1] - candidates[:, 0]) + (target[1] - target[0]) - inter
    return inter / np.clip(union, 1e-8, None)


def greedy_nms(segments, scores, iou_thresh):
    """Greedy NMS sorted by descending score."""
    if len(segments) == 0:
        return np.empty((0, 2)), np.empty(0)
    order = np.argsort(-scores)
    segments = segments[order]
    scores = scores[order]
    keep_segs, keep_scores = [], []
    for i in range(len(segments)):
        seg = segments[i]
        if len(keep_segs) == 0:
            keep_segs.append(seg)
            keep_scores.append(scores[i])
            continue
        ious = segment_iou(seg, np.array(keep_segs))
        if ious.max() < iou_thresh:
            keep_segs.append(seg)
            keep_scores.append(scores[i])
    return np.array(keep_segs) if keep_segs else np.empty((0, 2)), np.array(keep_scores)


def merge_intervals(intervals):
    """Merge overlapping intervals."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def compute_operating_curve(result_dict, ann_data, subset, tiou_match):
    """
    Compute precision/recall curve and collect boundary errors for TP detections.
    """
    gt_per_clip = {}
    n_gt_total = 0
    for clip_name, clip_info in ann_data.items():
        if clip_info.get("subset", "") != subset:
            continue
        gt_segs = np.array(
            [[float(a["segment"][0]), float(a["segment"][1])]
             for a in clip_info.get("annotations", [])],
            dtype=np.float64,
        ).reshape(-1, 2)
        gt_per_clip[clip_name] = gt_segs
        n_gt_total += len(gt_segs)

    all_scores = []
    all_is_tp = []
    all_boundary_errors = []

    for clip_name, predictions in result_dict.items():
        if clip_name not in gt_per_clip:
            continue
        gt_segs = gt_per_clip[clip_name]
        n_gt = len(gt_segs)

        preds_sorted = sorted(predictions, key=lambda x: x["score"], reverse=True)
        gt_matched = np.zeros(n_gt, dtype=bool)

        for pred in preds_sorted:
            score = pred["score"]
            pred_seg = np.array([pred["segment"][0], pred["segment"][1]])
            is_tp = False

            if n_gt > 0:
                ious = segment_iou(pred_seg, gt_segs)
                sorted_gt_idx = np.argsort(-ious)
                for gt_idx in sorted_gt_idx:
                    if ious[gt_idx] < tiou_match:
                        break
                    if not gt_matched[gt_idx]:
                        is_tp = True
                        gt_matched[gt_idx] = True
                        gt_seg = gt_segs[gt_idx]
                        left_err = pred_seg[0] - gt_seg[0]
                        right_err = gt_seg[1] - pred_seg[1]
                        all_boundary_errors.append((left_err, right_err))
                        break

            all_scores.append(score)
            all_is_tp.append(is_tp)

    all_scores = np.array(all_scores)
    all_is_tp = np.array(all_is_tp, dtype=bool)

    if len(all_scores) == 0:
        return None, np.array([]).reshape(0, 2)

    # Build curve at many thresholds
    unique_scores = np.unique(all_scores)
    linear_grid = np.linspace(all_scores.min(), all_scores.max(), 2000)
    thresholds = np.sort(np.unique(np.concatenate([unique_scores, linear_grid])))

    precisions, recalls, f2_scores, n_accepted_list = [], [], [], []

    for thresh in thresholds:
        mask = all_scores >= thresh
        n_above = mask.sum()
        n_tp_above = all_is_tp[mask].sum() if n_above > 0 else 0

        prec = 1.0 if n_above == 0 else n_tp_above / n_above
        rec = 0.0 if n_above == 0 else n_tp_above / max(n_gt_total, 1)

        f2 = (5 * prec * rec / (4 * prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f2_scores.append(f2)
        n_accepted_list.append(n_above)

    curve = {
        'thresholds': thresholds,
        'precision': np.array(precisions),
        'recall': np.array(recalls),
        'f2': np.array(f2_scores),
        'n_accepted': np.array(n_accepted_list),
        'n_gt_total': n_gt_total,
    }
    return curve, np.array(all_boundary_errors)


def find_threshold_for_recall(curve, target_recall, min_precision=0.0):
    """Find highest threshold achieving >= target_recall."""
    thresholds = curve['thresholds']
    recalls = curve['recall']
    precisions = curve['precision']

    valid_mask = (recalls >= target_recall) & (precisions >= min_precision)

    if not valid_mask.any():
        best_idx = int(np.argmax(recalls))
        achieved = False
    else:
        valid_indices = np.where(valid_mask)[0]
        best_idx = int(valid_indices[np.argmax(thresholds[valid_indices])])
        achieved = True

    return {
        'threshold': float(thresholds[best_idx]),
        'recall': float(recalls[best_idx]),
        'precision': float(precisions[best_idx]),
        'f2': float(curve['f2'][best_idx]),
        'n_accepted': int(curve['n_accepted'][best_idx]),
        'achieved': achieved,
    }


def estimate_padding(boundary_errors, percentile=90):
    """Estimate padding from boundary errors of TP detections."""
    if len(boundary_errors) == 0:
        return 15.0, {}

    left_errors = boundary_errors[:, 0]
    right_errors = boundary_errors[:, 1]

    left_positive = left_errors[left_errors > 0]
    right_positive = right_errors[right_errors > 0]

    stats = {
        'n_tp_total': len(boundary_errors),
        'n_left_positive': len(left_positive),
        'n_right_positive': len(right_positive),
        'pct_left_positive': len(left_positive) / max(len(boundary_errors), 1) * 100,
        'pct_right_positive': len(right_positive) / max(len(boundary_errors), 1) * 100,
    }

    if len(left_positive) > 0:
        left_pad = np.percentile(left_positive, percentile)
        stats['left_median'] = float(np.median(left_positive))
        stats['left_p75'] = float(np.percentile(left_positive, 75))
        stats['left_p90'] = float(np.percentile(left_positive, 90))
        stats['left_p95'] = float(np.percentile(left_positive, 95))
        stats['left_max'] = float(left_positive.max())
    else:
        left_pad = 5.0

    if len(right_positive) > 0:
        right_pad = np.percentile(right_positive, percentile)
        stats['right_median'] = float(np.median(right_positive))
        stats['right_p75'] = float(np.percentile(right_positive, 75))
        stats['right_p90'] = float(np.percentile(right_positive, 90))
        stats['right_p95'] = float(np.percentile(right_positive, 95))
        stats['right_max'] = float(right_positive.max())
    else:
        right_pad = 5.0

    recommended = max(left_pad, right_pad)
    stats['recommended_padding'] = float(recommended)
    return recommended, stats


def compute_review_time(result_dict, ann_data, subset, threshold, nms_iou, padding_sec):
    """
    Compute total review time after filtering, NMS, padding, and merging.
    """
    total_video_time = 0.0
    total_review_time = 0.0
    n_clips_with_detections = 0
    n_clips_total = 0
    n_clips_skipped = 0

    for clip_name, clip_info in ann_data.items():
        if clip_info.get("subset", "") != subset:
            continue

        duration = float(clip_info.get("duration", 120.0))
        total_video_time += duration
        n_clips_total += 1

        predictions = result_dict.get(clip_name, [])
        filtered = [(p["segment"][0], p["segment"][1], p["score"])
                    for p in predictions if p["score"] >= threshold]

        if not filtered:
            n_clips_skipped += 1
            continue

        segs = np.array([(s, e) for s, e, _ in filtered])
        scores = np.array([sc for _, _, sc in filtered])
        nms_segs, _ = greedy_nms(segs, scores, nms_iou)

        if len(nms_segs) == 0:
            n_clips_skipped += 1
            continue

        n_clips_with_detections += 1

        padded = []
        for seg in nms_segs:
            start = max(0.0, seg[0] - padding_sec)
            end = min(duration, seg[1] + padding_sec)
            padded.append((start, end))

        merged = merge_intervals(padded)
        clip_review = sum(e - s for s, e in merged)
        total_review_time += min(clip_review, duration)

    return {
        'total_video_time_sec': total_video_time,
        'total_review_time_sec': total_review_time,
        'total_video_time_hours': total_video_time / 3600,
        'total_review_time_hours': total_review_time / 3600,
        'time_saved_hours': (total_video_time - total_review_time) / 3600,
        'reduction_percent': (1 - total_review_time / max(total_video_time, 1)) * 100,
        'n_clips_total': n_clips_total,
        'n_clips_with_detections': n_clips_with_detections,
        'n_clips_skipped': n_clips_skipped,
        'avg_review_per_active_clip_sec': total_review_time / max(n_clips_with_detections, 1),
    }


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # DDP init
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.rank = int(os.environ.get("RANK", 0))

    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    set_seed(args.seed)

    cfg = update_workdir(cfg, args.id, args.world_size)
    logger = setup_logger("ReviewTime", save_dir=cfg.work_dir, distributed_rank=args.rank)

    # ─── Build model once (reused for both subsets) ───
    need_model = (args.load_predictions is None) or (
        args.load_val_predictions is None and args.fixed_padding is None
    )
    model = None
    if need_model:
        logger.info("Building model...")
        model = build_model_and_load(cfg, args, logger)

    # ─── Get TEST predictions ───
    if args.load_predictions is not None:
        logger.info(f"Loading test predictions from: {args.load_predictions}")
        with open(args.load_predictions, "r") as f:
            pred_data = json.load(f)
        test_result_dict = pred_data.get("results", pred_data)
    else:
        logger.info(f"Running inference on '{args.subset}' subset...")
        test_result_dict = run_inference_on_subset(cfg, args, model, args.subset, logger)

        if args.save_predictions and args.rank == 0:
            save_path = os.path.join(cfg.work_dir, f"result_detection_{args.subset}.json")
            with open(save_path, "w") as f:
                json.dump({"results": test_result_dict}, f)
            logger.info(f"Test predictions saved to: {save_path}")

    # ─── Get VALIDATION predictions (for padding estimation) ───
    val_result_dict = None
    if args.fixed_padding is None:
        if args.padding_subset == args.subset:
            # Same subset — reuse (user's choice, but warn)
            logger.info(f"NOTE: padding_subset == subset ('{args.subset}'). "
                        f"Using same data for padding estimation.")
            val_result_dict = test_result_dict
        elif args.load_val_predictions is not None:
            logger.info(f"Loading validation predictions from: {args.load_val_predictions}")
            with open(args.load_val_predictions, "r") as f:
                val_pred_data = json.load(f)
            val_result_dict = val_pred_data.get("results", val_pred_data)
        else:
            logger.info(f"Running inference on '{args.padding_subset}' subset (for padding)...")
            val_result_dict = run_inference_on_subset(
                cfg, args, model, args.padding_subset, logger
            )

            if args.save_predictions and args.rank == 0:
                save_path = os.path.join(
                    cfg.work_dir, f"result_detection_{args.padding_subset}.json"
                )
                with open(save_path, "w") as f:
                    json.dump({"results": val_result_dict}, f)
                logger.info(f"Validation predictions saved to: {save_path}")

    # Only rank 0 does the analysis
    if args.rank != 0:
        dist.barrier()
        return

    # ─── Load annotations ───
    ann_path = cfg.evaluation.ground_truth_filename
    logger.info(f"Loading annotations: {ann_path}")
    with open(ann_path, "r") as f:
        ann_data = json.load(f).get("database", {})

    subset = args.subset
    subset_clips = [k for k, v in ann_data.items() if v.get("subset") == subset]
    total_duration = sum(float(ann_data[c].get("duration", 120)) for c in subset_clips)

    logger.info(f"Subset '{subset}': {len(subset_clips)} clips, "
                f"{total_duration/3600:.1f} hours total")

    # ─── Compute boundary errors from VALIDATION set ───
    if args.fixed_padding is not None:
        recommended_padding = args.fixed_padding
        pad_stats = None
        val_boundary_errors = np.array([]).reshape(0, 2)
        logger.info(f"Using fixed padding override: {recommended_padding:.1f}s")
    else:
        padding_subset = args.padding_subset
        logger.info(f"Computing boundary errors from '{padding_subset}' subset...")

        val_clips = [k for k, v in ann_data.items() if v.get("subset") == padding_subset]
        val_duration = sum(float(ann_data[c].get("duration", 120)) for c in val_clips)
        logger.info(f"  Padding subset '{padding_subset}': {len(val_clips)} clips, "
                    f"{val_duration/3600:.1f} hours")

        _, val_boundary_errors = compute_operating_curve(
            val_result_dict, ann_data, padding_subset, args.tiou_match
        )

        recommended_padding, pad_stats = estimate_padding(
            val_boundary_errors, percentile=args.padding_percentile
        )

    # ─── Compute operating curve on TEST set ───
    logger.info(f"Computing operating curve on '{subset}' (tIoU >= {args.tiou_match})...")
    curve, test_boundary_errors = compute_operating_curve(
        test_result_dict, ann_data, subset, args.tiou_match
    )

    if curve is None:
        logger.info("ERROR: No predictions found for the specified subset.")
        return

    logger.info(f"  Total GT events: {curve['n_gt_total']}")
    logger.info(f"  Total predictions evaluated: {len(curve['thresholds'])}")
    max_recall = curve['recall'].max()
    logger.info(f"  Maximum achievable recall: {max_recall*100:.1f}%")

    # ─── Find threshold ───
    if args.threshold_override is not None:
        threshold = args.threshold_override
        idx = np.searchsorted(curve['thresholds'], threshold)
        idx = min(idx, len(curve['thresholds']) - 1)
        recall_info = {
            'threshold': threshold,
            'recall': float(curve['recall'][idx]),
            'precision': float(curve['precision'][idx]),
            'f2': float(curve['f2'][idx]),
            'n_accepted': int(curve['n_accepted'][idx]),
            'achieved': True,
        }
    else:
        recall_info = find_threshold_for_recall(curve, args.target_recall)
        threshold = recall_info['threshold']

    # ─── Print operating point ───
    print(f"\n{'═' * 70}")
    print(f"OPERATING POINT SELECTION (on '{subset}' set)")
    print(f"{'═' * 70}")
    print(f"  Target recall:      >= {args.target_recall*100:.0f}%")
    print(f"  tIoU match:         >= {args.tiou_match}")
    print(f"  Selected threshold: {threshold:.4f}")
    print(f"  Achieved recall:    {recall_info['recall']*100:.1f}%")
    print(f"  Precision at OP:    {recall_info['precision']*100:.1f}%")
    print(f"  F2 score:           {recall_info['f2']:.3f}")
    print(f"  Detections above:   {recall_info['n_accepted']}")
    if not recall_info['achieved']:
        print(f"  ⚠ WARNING: {args.target_recall*100:.0f}% recall NOT achievable!")
        print(f"    Max recall = {max_recall*100:.1f}%. Showing best available.")

    # ─── Boundary error analysis → padding (from VALIDATION) ───
    print(f"\n{'═' * 70}")
    print(f"BOUNDARY ERROR ANALYSIS (from '{args.padding_subset}' set → padding estimation)")
    print(f"{'═' * 70}")

    if args.fixed_padding is not None:
        print(f"  Using fixed padding override: {recommended_padding:.1f}s")
    elif pad_stats:
        print(f"  Source: '{args.padding_subset}' subset (held-out from test)")
        print(f"  TP detections analyzed: {pad_stats['n_tp_total']}")
        print(f"")
        print(f"  LEFT boundary (pred starts AFTER GT start → misses event start):")
        print(f"    Occurs in {pad_stats['pct_left_positive']:.1f}% of TPs")
        if 'left_median' in pad_stats:
            print(f"    Median: {pad_stats['left_median']:.1f}s | "
                  f"75th: {pad_stats['left_p75']:.1f}s | "
                  f"90th: {pad_stats['left_p90']:.1f}s | "
                  f"95th: {pad_stats['left_p95']:.1f}s | "
                  f"Max: {pad_stats['left_max']:.1f}s")
        print(f"")
        print(f"  RIGHT boundary (pred ends BEFORE GT end → misses event end):")
        print(f"    Occurs in {pad_stats['pct_right_positive']:.1f}% of TPs")
        if 'right_median' in pad_stats:
            print(f"    Median: {pad_stats['right_median']:.1f}s | "
                  f"75th: {pad_stats['right_p75']:.1f}s | "
                  f"90th: {pad_stats['right_p90']:.1f}s | "
                  f"95th: {pad_stats['right_p95']:.1f}s | "
                  f"Max: {pad_stats['right_max']:.1f}s")
        print(f"")
        print(f"  ► Recommended padding ({args.padding_percentile:.0f}th pctl): "
              f"{recommended_padding:.1f}s per side")
        print(f"    Rationale: covers {args.padding_percentile:.0f}% of boundary-miss cases")
        print(f"    observed on the validation set.")
    else:
        print(f"  No TP data on '{args.padding_subset}'. Using default: {recommended_padding:.1f}s")

    # ─── Review time at recommended padding ───
    print(f"\n{'═' * 70}")
    print(f"REVIEW TIME ANALYSIS (padding = ±{recommended_padding:.1f}s)")
    print(f"{'═' * 70}")

    r_rec = compute_review_time(
        test_result_dict, ann_data, subset, threshold, args.nms_iou, recommended_padding
    )

    print(f"  Total video:            {r_rec['total_video_time_hours']:.2f} h "
          f"({r_rec['n_clips_total']} clips)")
    print(f"  Ethologist must review: {r_rec['total_review_time_hours']:.2f} h")
    print(f"  Time saved:             {r_rec['time_saved_hours']:.2f} h")
    print(f"  Reduction:              {r_rec['reduction_percent']:.1f}%")
    print(f"")
    print(f"  Clips with detections:  {r_rec['n_clips_with_detections']} / {r_rec['n_clips_total']}")
    print(f"  Clips skipped entirely: {r_rec['n_clips_skipped']} "
          f"({r_rec['n_clips_skipped']/max(r_rec['n_clips_total'],1)*100:.1f}%)")
    print(f"  Avg review / active clip: {r_rec['avg_review_per_active_clip_sec']:.1f}s")

    # ─── Padding comparison table ───
    print(f"\n{'═' * 70}")
    print(f"PADDING COMPARISON TABLE (threshold = {threshold:.4f})")
    print(f"{'═' * 70}")
    header = (f"  {'Padding':>8} | {'Review':>8} | {'Saved':>8} | "
              f"{'Reduction':>9} | {'Clips Active':>12} | {'Skipped':>8}")
    print(header)
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*12}-+-{'-'*8}")

    all_paddings = sorted(set(args.padding_candidates + [recommended_padding]))
    for pad in all_paddings:
        r = compute_review_time(test_result_dict, ann_data, subset, threshold, args.nms_iou, pad)
        marker = " *" if abs(pad - recommended_padding) < 0.01 else "  "
        skip_pct = r['n_clips_skipped'] / max(r['n_clips_total'], 1) * 100
        print(f"{marker}{pad:>6.1f}s | {r['total_review_time_hours']:>6.2f}h | "
              f"{r['time_saved_hours']:>6.2f}h | "
              f"{r['reduction_percent']:>8.1f}% | "
              f"{r['n_clips_with_detections']:>12} | "
              f"{skip_pct:>7.1f}%")

    print(f"  (* = recommended from '{args.padding_subset}' set, "
          f"{args.padding_percentile:.0f}th percentile)")

    # ─── What gets missed ───
    print(f"\n{'═' * 70}")
    print(f"WHAT THE ETHOLOGIST MISSES")
    print(f"{'═' * 70}")
    n_gt = curve['n_gt_total']
    actual_recall = recall_info['recall']
    n_caught = int(round(n_gt * actual_recall))
    n_missed = n_gt - n_caught
    print(f"  GT events in '{subset}': {n_gt}")
    print(f"  Detected (tIoU >= {args.tiou_match}): {n_caught} ({actual_recall*100:.1f}%)")
    print(f"  Missed: {n_missed}")
    if len(subset_clips) > 0:
        print(f"  ≈ {n_missed / (total_duration/3600):.1f} missed events per hour")
    print(f"")
    print(f"  Note: 'Missed' means the GT event is not overlapped by ANY")
    print(f"  accepted prediction at tIoU >= {args.tiou_match}. The ethologist")
    print(f"  would need to manually scan to find these.")

    # ─── Summary ───
    clip_dur_min = float(ann_data[subset_clips[0]].get("duration", 120)) / 60
    print(f"\n{'═' * 70}")
    print(f"PUBLICATION SUMMARY")
    print(f"{'═' * 70}")
    print(f"  Dataset:          {len(subset_clips)} × {clip_dur_min:.0f}-min clips "
          f"= {total_duration/3600:.1f}h")
    print(f"  Padding source:   '{args.padding_subset}' set boundary errors "
          f"({args.padding_percentile:.0f}th pctl)")
    print(f"  Operating point:  score ≥ {threshold:.3f} → "
          f"{actual_recall*100:.0f}% recall @ tIoU ≥ {args.tiou_match}")
    print(f"  Padding:          ±{recommended_padding:.0f}s per detection")
    print(f"  Review time:      {r_rec['total_review_time_hours']:.1f}h of "
          f"{r_rec['total_video_time_hours']:.1f}h "
          f"({r_rec['reduction_percent']:.0f}% reduction)")
    print(f"  Missed events:    {n_missed}/{n_gt} ({(1-actual_recall)*100:.0f}%)")
    print(f"{'═' * 70}")

    # ─── Save JSON output ───
    output = {
        'config': {
            'subset': subset,
            'padding_subset': args.padding_subset,
            'target_recall': args.target_recall,
            'tiou_match': args.tiou_match,
            'nms_iou': args.nms_iou,
            'padding_percentile': args.padding_percentile,
            'annotation_path': ann_path,
        },
        'operating_point': recall_info,
        'padding': {
            'recommended_sec': recommended_padding,
            'source_subset': args.padding_subset,
            'method': f"{args.padding_percentile}th_percentile_boundary_error_on_{args.padding_subset}",
        },
        'review_time_at_recommended_padding': {
            'total_video_hours': r_rec['total_video_time_hours'],
            'review_hours': r_rec['total_review_time_hours'],
            'saved_hours': r_rec['time_saved_hours'],
            'reduction_percent': r_rec['reduction_percent'],
            'clips_total': r_rec['n_clips_total'],
            'clips_with_detections': r_rec['n_clips_with_detections'],
            'clips_skipped': r_rec['n_clips_skipped'],
        },
        'events': {
            'gt_total': n_gt,
            'detected': n_caught,
            'missed': n_missed,
            'recall_pct': actual_recall * 100,
        },
        'padding_comparison': {},
    }
    for pad in all_paddings:
        r = compute_review_time(test_result_dict, ann_data, subset, threshold, args.nms_iou, pad)
        output['padding_comparison'][f'{pad:.1f}s'] = {
            'review_hours': round(r['total_review_time_hours'], 3),
            'reduction_percent': round(r['reduction_percent'], 1),
        }

    out_path = os.path.join(cfg.work_dir, "review_time_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    dist.barrier()


if __name__ == "__main__":
    main()