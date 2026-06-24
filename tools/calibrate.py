"""
Post-hoc calibration via detection operating curve + threshold selection.

This script:
1. Runs inference on the VALIDATION set
2. Labels each detection as TP/FP at multiple tIoU thresholds
3. Computes the full precision-recall operating curve
4. Selects thresholds for multiple operating points:
   a. Max-F2 (recall-weighted, best for screening)
   b. Target recall (e.g., >=70%, >=80% of events caught)
   c. Min period count MAE (for trend analysis)
5. Optionally fits isotonic calibration for score interpretability
6. Saves calibration.json with all operating points

Usage:
    torchrun --nnodes=1 --nproc_per_node=1 \
        --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
        tools/calibrate.py CONFIG \
        --checkpoint PATH \
        [--tiou_match 0.3] \
        [--target_recalls 0.7 0.8 0.9] \
        [--nms_iou 0.3] \
        [--output calibration.json]
"""

import os
import sys
import copy
import json
import argparse

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from torch.nn.parallel import DistributedDataParallel
from mmengine.config import Config, DictAction

from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader
from opentad.utils import update_workdir, set_seed, create_folder, setup_logger

from opentad.calibration.operating_curve import (
    label_detections,
    compute_detection_operating_curve,
    select_threshold_by_target_recall,
    select_threshold_by_max_f2,
)
from opentad.calibration.threshold_selection import (
    optimize_threshold_for_period_counts,
    compute_period_counts_at_threshold,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-hoc calibration for single-class TAD"
    )
    parser.add_argument("config", metavar="FILE", type=str)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument(
        "--tiou_match", type=float, default=0.3,
        help="tIoU threshold for labelling TP during calibration"
    )
    parser.add_argument(
        "--target_recalls", type=float, nargs="+", default=[0.6, 0.7, 0.8, 0.9],
        help="Target recall levels to find thresholds for"
    )
    parser.add_argument(
        "--nms_iou", type=float, default=0.3,
        help="tIoU for cross-clip NMS in period counting"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for calibration JSON"
    )
    parser.add_argument(
        "--use_isotonic", action="store_true",
        help="Also fit isotonic regression for score interpretability"
    )
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    args = parser.parse_args()
    return args


def run_inference(model, dataloader, cfg, rank, world_size, use_amp):
    """Run model inference and return result_dict."""
    from opentad.cores.test_engine import gather_ddp_results
    from opentad.models.utils.post_processing import build_classifier
    from opentad.datasets.base import SlidingWindowDataset

    cfg.inference["folder"] = os.path.join(cfg.work_dir, "calibration_outputs")
    cfg.inference["save_raw_prediction"] = False

    external_cls = None
    if "external_cls" in cfg.post_processing:
        ext_cls_cfg = cfg.post_processing.external_cls
        if ext_cls_cfg is not None:
            if isinstance(ext_cls_cfg, (list, tuple)):
                external_cls = list(ext_cls_cfg)
            else:
                external_cls = build_classifier(ext_cls_cfg)

    cfg.post_processing.sliding_window = isinstance(
        dataloader.dataset, SlidingWindowDataset
    )

    model.eval()
    result_dict = {}
    for data_dict in tqdm.tqdm(dataloader, disable=(rank != 0)):
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

    result_dict = gather_ddp_results(world_size, result_dict, cfg.post_processing)
    return result_dict

# Added visualisation

def generate_calibration_figure(
    curves,
    label_results,
    period_result,
    f2_result,
    recall_results,
    primary_tiou,
    work_dir,
    logger,
):
    """
    Generate a multi-panel figure illustrating the calibration process.

    Panel A: Score distributions of TP vs FP detections (histogram)
    Panel B: Precision-Recall operating curve with operating points marked
    Panel C: F2 score and Recall vs threshold
    Panel D: Period count MAE vs threshold

    Saves to {work_dir}/calibration_figure.pdf and .png
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch
    except ImportError:
        logger.warning("matplotlib not installed; skipping figure generation.")
        logger.warning("Install with: pip install matplotlib")
        return

    primary_scores = label_results[primary_tiou]['scores']
    primary_is_tp = label_results[primary_tiou]['is_tp']
    n_gt = label_results[primary_tiou]['n_gt']
    curve = curves[primary_tiou]

    tp_scores = primary_scores[primary_is_tp]
    fp_scores = primary_scores[~primary_is_tp]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"Detection Operating Curve Calibration (tIoU match $\\geq$ {primary_tiou})",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # ─── Panel A: Score distributions ───
    ax = axes[0, 0]
    bins = np.linspace(0, max(primary_scores.max(), 0.5), 60)

    ax.hist(
        fp_scores, bins=bins, alpha=0.6, color="#d62728", label=f"FP (n={len(fp_scores)})",
        density=True, edgecolor="none",
    )
    ax.hist(
        tp_scores, bins=bins, alpha=0.7, color="#2ca02c", label=f"TP (n={len(tp_scores)})",
        density=True, edgecolor="none",
    )

    # Mark key thresholds
    thresh_f2 = f2_result['threshold']
    thresh_count = period_result['optimal_threshold']
    ymax = ax.get_ylim()[1]

    ax.axvline(thresh_f2, color="#1f77b4", linestyle="--", linewidth=1.5,
               label=f"Max-F2 threshold ({thresh_f2:.3f})")
    ax.axvline(thresh_count, color="#ff7f0e", linestyle=":", linewidth=1.5,
               label=f"Count-MAE threshold ({thresh_count:.3f})")

    ax.set_xlabel("Raw detection score")
    ax.set_ylabel("Density")
    ax.set_title("A. Score distributions of TP vs FP detections")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, max(0.5, primary_scores.max() * 1.05))

    # ─── Panel B: Precision-Recall curve ───
    ax = axes[0, 1]

    # Plot the PR curve
    recall = curve['recall']
    precision = curve['precision']
    ax.plot(recall, precision, color="#1f77b4", linewidth=1.5, label=f"tIoU $\\geq$ {primary_tiou}")

    # If we have a second tIoU curve, plot it too
    for tiou_val, c in curves.items():
        if tiou_val != primary_tiou:
            ax.plot(c['recall'], c['precision'], color="#9467bd", linewidth=1.2,
                    linestyle="--", alpha=0.7, label=f"tIoU $\\geq$ {tiou_val}")

    # Mark operating points
    ax.scatter(
        [f2_result['recall']], [f2_result['precision']],
        color="#1f77b4", s=100, zorder=5, marker="*",
        label=f"Max-F2 (R={f2_result['recall']:.2f}, P={f2_result['precision']:.2f})",
    )

    for target_rec, res in recall_results.items():
        if res['achieved']:
            ax.scatter(
                [res['recall']], [res['precision']],
                color="#2ca02c", s=60, zorder=5, marker="o",
                label=f"Target R$\\geq${target_rec:.0%} (t={res['threshold']:.3f})",
            )

    ax.set_xlabel("Recall (fraction of GT events detected)")
    ax.set_ylabel("Precision (fraction of predictions that are TP)")
    ax.set_title("B. Precision-Recall operating curve")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # ─── Panel C: F2 and Recall vs threshold ───
    ax = axes[1, 0]

    thresholds = curve['thresholds']
    ax.plot(thresholds, curve['recall'], color="#2ca02c", linewidth=1.5, label="Recall")
    ax.plot(thresholds, curve['precision'], color="#d62728", linewidth=1.5, label="Precision")
    ax.plot(thresholds, curve['f2'], color="#1f77b4", linewidth=2.0, label="F2 score")
    ax.plot(thresholds, curve['f1'], color="#9467bd", linewidth=1.2, linestyle="--",
            alpha=0.7, label="F1 score")

    # Mark max-F2 point
    ax.axvline(thresh_f2, color="#1f77b4", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.scatter([thresh_f2], [f2_result['f2']], color="#1f77b4", s=80, zorder=5, marker="*")

    # Mark count-MAE threshold
    ax.axvline(thresh_count, color="#ff7f0e", linestyle=":", linewidth=1.0, alpha=0.7)

    ax.set_xlabel("Score threshold")
    ax.set_ylabel("Metric value")
    ax.set_title("C. Recall, Precision, F1, and F2 vs score threshold")
    ax.set_xlim(0, max(0.5, thresholds.max() * 0.8))
    ax.set_ylim(0, 1.02)
    ax.legend(loc="right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ─── Panel D: Period count MAE vs threshold ───
    ax = axes[1, 1]

    all_thresh = np.array(period_result['all_thresholds'])
    all_maes = np.array(period_result['all_maes'])

    ax.plot(all_thresh, all_maes, color="#ff7f0e", linewidth=1.5)
    ax.axvline(thresh_count, color="#ff7f0e", linestyle=":", linewidth=1.5,
               label=f"Optimal ({thresh_count:.3f}, MAE={period_result['optimal_mae']:.2f})")
    ax.scatter([thresh_count], [period_result['optimal_mae']],
               color="#ff7f0e", s=80, zorder=5, marker="D")

    # Also mark the F2 threshold on this plot
    f2_idx = np.argmin(np.abs(all_thresh - thresh_f2))
    if f2_idx < len(all_maes):
        ax.axvline(thresh_f2, color="#1f77b4", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.scatter([thresh_f2], [all_maes[f2_idx]], color="#1f77b4", s=60, zorder=5, marker="*",
                   label=f"Max-F2 point (MAE={all_maes[f2_idx]:.2f})")

    ax.set_xlabel("Score threshold")
    ax.set_ylabel("Period count MAE (events)")
    ax.set_title("D. Period-level count MAE vs score threshold")
    ax.set_xlim(0, max(0.5, all_thresh.max() * 0.8))
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ─── Layout and save ───
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    for ext in ["pdf", "png"]:
        out_path = os.path.join(work_dir, f"calibration_figure.{ext}")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Figure saved: {out_path}")

    plt.close(fig)


def generate_score_threshold_diagram(
    result_dict,
    annotation_path,
    subset,
    primary_tiou,
    label_results,
    f2_result,
    period_result,
    nms_iou,
    work_dir,
    logger,
):
    """
    Generate a schematic diagram illustrating the full inference pipeline:
    clip-level predictions -> absolute time conversion -> cross-clip NMS ->
    thresholding -> period counts.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
        import json
    except ImportError:
        logger.warning("matplotlib not installed; skipping diagram generation.")
        return

    with open(annotation_path, "r") as f:
        ann_data = json.load(f).get("database", {})

    import re

    def get_period(clip_name):
        match = re.match(r"(.+)_clip\d+$", clip_name)
        return match.group(1) if match else clip_name

    # Find a period with multiple clips and some predictions to illustrate
    period_clips = {}
    for clip_name, clip_info in ann_data.items():
        if subset is not None and clip_info.get("subset", "") != subset:
            continue
        period = get_period(clip_name)
        if period not in period_clips:
            period_clips[period] = []
        period_clips[period].append((clip_name, clip_info))

    # Pick a period with at least 2 clips and some GT events
    example_period = None
    for period, clips in sorted(period_clips.items()):
        total_gt = sum(len(c[1].get("annotations", [])) for c in clips)
        if len(clips) >= 3 and total_gt >= 2:
            example_period = period
            break

    if example_period is None:
        # Fallback: just pick first period with multiple clips
        for period, clips in sorted(period_clips.items()):
            if len(clips) >= 2:
                example_period = period
                break

    if example_period is None:
        logger.info("  Could not find suitable example period for diagram.")
        return

    clips = sorted(period_clips[example_period], key=lambda x: x[1].get("source_start", 0))
    # Limit to first 4 clips for readability
    clips = clips[:4]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 2, 1.5, 1.5]})
    fig.suptitle(
        f"Inference Pipeline Illustration: {example_period}",
        fontsize=12, fontweight="bold", y=0.98,
    )

    # Determine time range
    min_time = min(c[1].get("source_start", 0) for c in clips)
    max_time = max(c[1].get("source_start", 0) + c[1].get("duration", 300) for c in clips)
    time_span = max_time - min_time

    threshold = f2_result['threshold']
    colors_clip = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # ─── Panel 1: Clip layout and GT segments ───
    ax = axes[0]
    ax.set_title("Step 1: Clips with ground truth events (absolute time)", fontsize=10, pad=8)

    for i, (clip_name, clip_info) in enumerate(clips):
        start = clip_info.get("source_start", 0)
        duration = clip_info.get("duration", 300)
        color = colors_clip[i % len(colors_clip)]

        # Draw clip extent
        rect = Rectangle((start, i * 1.2), duration, 0.8, linewidth=1.5,
                         edgecolor=color, facecolor=color, alpha=0.15)
        ax.add_patch(rect)
        ax.text(start + 5, i * 1.2 + 0.85, clip_name.split("_")[-1],
                fontsize=7, color=color, fontweight="bold", va="bottom")

        # Draw GT segments
        for ann in clip_info.get("annotations", []):
            gt_start = start + float(ann["segment"][0])
            gt_end = start + float(ann["segment"][1])
            gt_rect = Rectangle((gt_start, i * 1.2 + 0.1), gt_end - gt_start, 0.6,
                                linewidth=0, facecolor="#2ca02c", alpha=0.6)
            ax.add_patch(gt_rect)

    ax.set_xlim(min_time - 10, max_time + 10)
    ax.set_ylim(-0.2, len(clips) * 1.2 + 0.5)
    ax.set_xlabel("Absolute time (seconds)")
    ax.set_yticks([])
    # Legend
    ax.plot([], [], color="#2ca02c", linewidth=8, alpha=0.6, label="Ground truth event")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)

    # ─── Panel 2: Predictions per clip (clip-local converted to absolute) ───
    ax = axes[1]
    ax.set_title(
        f"Step 2: Predictions converted to absolute time (colour = score)",
        fontsize=10, pad=8,
    )

    all_abs_preds = []  # (abs_start, abs_end, score, clip_idx)
    for i, (clip_name, clip_info) in enumerate(clips):
        start = clip_info.get("source_start", 0)
        duration = clip_info.get("duration", 300)
        color = colors_clip[i % len(colors_clip)]

        # Draw clip extent
        rect = Rectangle((start, i * 1.2), duration, 0.8, linewidth=1.0,
                         edgecolor=color, facecolor="none", linestyle="--", alpha=0.5)
        ax.add_patch(rect)

        # Get predictions for this clip
        preds = result_dict.get(clip_name, [])
        for pred in preds:
            abs_start = start + pred["segment"][0]
            abs_end = start + pred["segment"][1]
            score = pred["score"]
            all_abs_preds.append((abs_start, abs_end, score, i))

            # Color by score intensity
            alpha = min(0.9, max(0.2, score * 2))
            if score >= threshold:
                pred_color = "#1f77b4"
            else:
                pred_color = "#aaaaaa"

            pred_rect = Rectangle(
                (abs_start, i * 1.2 + 0.1), abs_end - abs_start, 0.6,
                linewidth=0.5, edgecolor=pred_color, facecolor=pred_color, alpha=alpha * 0.5,
            )
            ax.add_patch(pred_rect)

    ax.set_xlim(min_time - 10, max_time + 10)
    ax.set_ylim(-0.2, len(clips) * 1.2 + 0.5)
    ax.set_xlabel("Absolute time (seconds)")
    ax.set_yticks([])
    ax.plot([], [], color="#1f77b4", linewidth=6, alpha=0.5, label=f"Above threshold ({threshold:.3f})")
    ax.plot([], [], color="#aaaaaa", linewidth=6, alpha=0.5, label="Below threshold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)

    # ─── Panel 3: After thresholding ───
    ax = axes[2]
    ax.set_title(
        f"Step 3: Predictions above threshold (score $\\geq$ {threshold:.3f})",
        fontsize=10, pad=8,
    )

    above_threshold = [(s, e, sc, ci) for s, e, sc, ci in all_abs_preds if sc >= threshold]
    above_threshold.sort(key=lambda x: x[2], reverse=True)

    for j, (s, e, sc, ci) in enumerate(above_threshold):
        alpha = min(0.9, max(0.3, sc))
        rect = Rectangle((s, 0.1), e - s, 0.8, linewidth=1.0,
                         edgecolor="#1f77b4", facecolor="#1f77b4", alpha=alpha * 0.6)
        ax.add_patch(rect)
        if j < 8:  # Label first few
            ax.text((s + e) / 2, 0.95, f"{sc:.2f}", fontsize=6, ha="center", va="bottom")

    ax.set_xlim(min_time - 10, max_time + 10)
    ax.set_ylim(0, 1.3)
    ax.set_xlabel("Absolute time (seconds)")
    ax.set_yticks([])
    ax.text(max_time + 15, 0.5, f"n={len(above_threshold)}", fontsize=9,
            ha="left", va="center", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    # ─── Panel 4: After NMS deduplication ───
    ax = axes[3]
    ax.set_title(
        f"Step 4: After cross-clip NMS (tIoU threshold = {nms_iou})",
        fontsize=10, pad=8,
    )

    # Apply greedy NMS
    def _temporal_iou(a, b):
        inter_s = max(a[0], b[0])
        inter_e = min(a[1], b[1])
        inter = max(0.0, inter_e - inter_s)
        union = (a[1] - a[0]) + (b[1] - b[0]) - inter
        return inter / max(union, 1e-8)

    keep = []
    for p in above_threshold:
        seg = (p[0], p[1])
        if not any(_temporal_iou(seg, (k[0], k[1])) >= nms_iou for k in keep):
            keep.append(p)

    # Draw GT for reference
    for clip_name, clip_info in clips:
        start = clip_info.get("source_start", 0)
        for ann in clip_info.get("annotations", []):
            gt_start = start + float(ann["segment"][0])
            gt_end = start + float(ann["segment"][1])
            gt_rect = Rectangle((gt_start, 0.05), gt_end - gt_start, 0.3,
                                linewidth=0, facecolor="#2ca02c", alpha=0.4)
            ax.add_patch(gt_rect)

    for j, (s, e, sc, ci) in enumerate(keep):
        rect = Rectangle((s, 0.45), e - s, 0.5, linewidth=1.5,
                         edgecolor="#1f77b4", facecolor="#1f77b4", alpha=0.6)
        ax.add_patch(rect)
        ax.text((s + e) / 2, 1.0, f"{sc:.2f}", fontsize=7, ha="center", va="bottom")

    # Count comparison
    gt_count = sum(len(c[1].get("annotations", [])) for c in clips)
    ax.set_xlim(min_time - 10, max_time + 10)
    ax.set_ylim(0, 1.3)
    ax.set_xlabel("Absolute time (seconds)")
    ax.set_yticks([])
    ax.text(
        max_time + 15, 0.7,
        f"Predicted: {len(keep)}\nGT: {gt_count}",
        fontsize=9, ha="left", va="center", fontweight="bold",
    )
    ax.plot([], [], color="#2ca02c", linewidth=6, alpha=0.4, label="Ground truth")
    ax.plot([], [], color="#1f77b4", linewidth=6, alpha=0.6, label="Retained predictions")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 0.95, 0.96])

    for ext in ["pdf", "png"]:
        out_path = os.path.join(work_dir, f"inference_pipeline_diagram.{ext}")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Diagram saved: {out_path}")

    plt.close(fig)

def main():
    args = parse_args()

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

    if args.rank == 0:
        create_folder(cfg.work_dir)

    logger = setup_logger("Calibrate", save_dir=cfg.work_dir, distributed_rank=args.rank)

    # ─── Build validation loader ───
    eval_val_cfg = copy.deepcopy(cfg.dataset.test)
    eval_val_cfg.subset_name = "validation"
    eval_val_dataset = build_dataset(eval_val_cfg, default_args=dict(logger=logger))
    eval_val_loader = build_dataloader(
        eval_val_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=False,
        drop_last=False,
        seed=args.seed,
        **cfg.solver.val,
    )

    # ─── Build model ───
    model = build_detector(cfg.model)
    model = model.to(args.local_rank)
    model = DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank
    )

    device = f"cuda:{args.local_rank}"
    checkpoint = torch.load(args.checkpoint, map_location=device)
    use_ema = getattr(cfg.solver, "ema", False)
    if use_ema and "state_dict_ema" in checkpoint:
        model.load_state_dict(checkpoint["state_dict_ema"])
        logger.info("Loaded EMA weights.")
    else:
        model.load_state_dict(checkpoint["state_dict"])
        logger.info("Loaded standard weights.")
    logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', '?')}")

    use_amp = getattr(cfg.solver, "amp", False)

    # ─── Step 1: Inference ───
    logger.info("\n" + "=" * 70)
    logger.info("DETECTION OPERATING CURVE CALIBRATION")
    logger.info("=" * 70)
    logger.info("\nStep 1: Running inference on validation set...")

    result_dict = run_inference(model, eval_val_loader, cfg, args.rank, args.world_size, use_amp)

    if args.rank != 0:
        return

    total_preds = sum(len(v) for v in result_dict.values())
    logger.info(f"  {len(result_dict)} clips, {total_preds} total detections")

    annotation_path = cfg.evaluation.ground_truth_filename

    # ─── Step 2: Label detections at multiple tIoU levels ───
    logger.info(f"\nStep 2: Labelling detections...")

    tiou_levels_for_curves = [0.3, 0.5]  # Main operating curves
    curves = {}
    label_results = {}

    for tiou_match in tiou_levels_for_curves:
        scores, is_tp, video_ids, n_gt, gt_per_video = label_detections(
            result_dict, annotation_path, subset="validation", tiou_match=tiou_match
        )
        n_tp = int(is_tp.sum())
        logger.info(
            f"  tIoU >= {tiou_match}: {n_tp} TP / {len(scores)} detections "
            f"({n_tp / max(len(scores), 1) * 100:.1f}% TP rate), "
            f"{n_gt} GT events total"
        )

        # Compute operating curve
        curve = compute_detection_operating_curve(scores, is_tp, n_gt)
        curves[tiou_match] = curve
        label_results[tiou_match] = {
            'scores': scores, 'is_tp': is_tp, 'n_gt': n_gt,
            'gt_per_video': gt_per_video,
        }

    # ─── Step 3: Select operating points ───
    logger.info(f"\nStep 3: Selecting operating points...")

    # Use tIoU=0.3 as primary (lenient matching appropriate for screening)
    primary_curve = curves[args.tiou_match if args.tiou_match in curves else 0.3]
    primary_tiou = args.tiou_match if args.tiou_match in curves else 0.3

    # A) Max F2 threshold
    f2_result = select_threshold_by_max_f2(primary_curve, min_recall=0.0)
    logger.info(f"\n  A) Max-F2 operating point (recall-weighted, best for screening):")
    logger.info(f"     Threshold: {f2_result['threshold']:.4f}")
    logger.info(f"     Recall:    {f2_result['recall'] * 100:.1f}%")
    logger.info(f"     Precision: {f2_result['precision'] * 100:.1f}%")
    logger.info(f"     F2:        {f2_result['f2'] * 100:.1f}%")

    # B) Target recall thresholds
    recall_results = {}
    logger.info(f"\n  B) Target recall operating points (tIoU >= {primary_tiou}):")
    logger.info(f"     {'Target':>8} {'Threshold':>10} {'Recall':>8} {'Precision':>10} {'F1':>6} {'Achieved':>9}")
    logger.info(f"     {'-' * 58}")

    for target_rec in sorted(args.target_recalls, reverse=True):
        res = select_threshold_by_target_recall(primary_curve, target_rec, min_precision=0.0)
        recall_results[target_rec] = res
        marker = "✓" if res['achieved'] else "✗"
        logger.info(
            f"     {target_rec * 100:>6.0f}%  {res['threshold']:>10.4f} "
            f"{res['recall'] * 100:>6.1f}%  {res['precision'] * 100:>8.1f}%  "
            f"{res['f1'] * 100:>4.1f}%  {marker:>8}"
        )

    # C) Period count MAE optimization
    logger.info(f"\n  C) Period count MAE optimization:")
    period_result = optimize_threshold_for_period_counts(
        result_dict, annotation_path, subset="validation",
        nms_iou_threshold=args.nms_iou, n_candidates=1000, #originally 1000 for all others
    )
    logger.info(f"     Optimal threshold: {period_result['optimal_threshold']:.4f}")
    logger.info(f"     Count MAE: {period_result['optimal_mae']:.2f} events/period")

    # Show period counts at the count-optimal threshold
    period_counts = compute_period_counts_at_threshold(
        result_dict, annotation_path, subset="validation",
        threshold=period_result['optimal_threshold'],
        nms_iou_threshold=args.nms_iou,
    )
    logger.info(f"\n     {'Period':<42} {'GT':>4} {'Pred':>4} {'Err':>4}")
    logger.info(f"     {'-' * 56}")
    for period in period_counts['periods']:
        gt_c = period_counts['period_gt'][period]
        pr_c = period_counts['period_pred'][period]
        err = abs(gt_c - pr_c)
        logger.info(f"     {period:<42} {gt_c:>4} {pr_c:>4} {err:>4}")

    # ─── Step 4: Score distribution analysis ───
    logger.info(f"\nStep 4: Score distribution analysis...")
    primary_scores = label_results[primary_tiou]['scores']
    primary_is_tp = label_results[primary_tiou]['is_tp']

    tp_scores = primary_scores[primary_is_tp]
    fp_scores = primary_scores[~primary_is_tp]

    logger.info(f"  TP scores: mean={tp_scores.mean():.4f}, "
                f"median={np.median(tp_scores):.4f}, "
                f"min={tp_scores.min():.4f}, max={tp_scores.max():.4f}")
    logger.info(f"  FP scores: mean={fp_scores.mean():.4f}, "
                f"median={np.median(fp_scores):.4f}, "
                f"min={fp_scores.min():.4f}, max={fp_scores.max():.4f}")

    # Percentiles of TP scores — shows where real events land
    if len(tp_scores) >= 5:
        percentiles = [10, 25, 50, 75, 90]
        vals = np.percentile(tp_scores, percentiles)
        logger.info(f"  TP score percentiles:")
        for p, v in zip(percentiles, vals):
            logger.info(f"    P{p:02d}: {v:.4f}")
        logger.info(f"  → To catch ≥90% of TPs, need threshold ≤ {vals[0]:.4f}")
        logger.info(f"  → To catch ≥75% of TPs, need threshold ≤ {vals[1]:.4f}")

    # ─── Step 5: Isotonic calibration (optional) ───
    isotonic_info = None
    if args.use_isotonic:
        logger.info(f"\nStep 5: Fitting isotonic regression for score interpretability...")
        try:
            from opentad.calibration.operating_curve import apply_isotonic_calibration
            cal_probs, calibrator = apply_isotonic_calibration(primary_scores, primary_is_tp)
            logger.info(f"  Isotonic regression fitted successfully.")
            logger.info(f"  Example mappings (raw → P(TP)):")
            example_scores = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
            for s in example_scores:
                p = calibrator(np.array([s]))[0]
                logger.info(f"    raw={s:.2f} → P(TP)={p:.4f}")
            isotonic_info = {
                'example_mappings': {str(s): float(calibrator(np.array([s]))[0]) for s in example_scores}
            }
        except ImportError:
            logger.warning("  sklearn not available; skipping isotonic regression.")
            logger.warning("  Install with: pip install scikit-learn")

    # ─── Step 6: Cross-check — what do the operating points mean for period counts? ───
    logger.info(f"\nStep 6: Period counts at each operating point...")
    operating_points = {
        'max_f2': f2_result['threshold'],
        'count_mae_optimal': period_result['optimal_threshold'],
    }
    for target_rec, res in recall_results.items():
        operating_points[f'recall_{int(target_rec * 100)}'] = res['threshold']

    logger.info(f"  {'Operating Point':<22} {'Threshold':>10} {'Total Pred':>10} "
                f"{'Total GT':>8} {'Period MAE':>10}")
    logger.info(f"  {'-' * 65}")

    for name, thresh in sorted(operating_points.items(), key=lambda x: x[1]):
        pc = compute_period_counts_at_threshold(
            result_dict, annotation_path, subset="validation",
            threshold=thresh, nms_iou_threshold=args.nms_iou,
        )
        logger.info(
            f"  {name:<22} {thresh:>10.4f} {pc['total_pred']:>10} "
            f"{pc['total_gt']:>8} {pc['mae']:>10.2f}"
        )

    # ─── Generate figures ───
    logger.info(f"\nGenerating calibration figures...")

    generate_calibration_figure(
        curves=curves,
        label_results=label_results,
        period_result=period_result,
        f2_result=f2_result,
        recall_results=recall_results,
        primary_tiou=primary_tiou,
        work_dir=cfg.work_dir,
        logger=logger,
    )

    generate_score_threshold_diagram(
        result_dict=result_dict,
        annotation_path=annotation_path,
        subset="validation",
        primary_tiou=primary_tiou,
        label_results=label_results,
        f2_result=f2_result,
        period_result=period_result,
        nms_iou=args.nms_iou,
        work_dir=cfg.work_dir,
        logger=logger,
    )

    # ─── Step 7: Recommended operating point ───
    # For ethological screening: use max-F2 as default, but warn about
    # validation size limitations
    n_gt_val = label_results[primary_tiou]['n_gt']
    logger.info(f"\n{'=' * 70}")
    logger.info("RECOMMENDATION")
    logger.info(f"{'=' * 70}")
    logger.info(f"  Validation GT events: {n_gt_val} (small sample — interpret with caution)")
    logger.info(f"")
    logger.info(f"  For SCREENING (catch most events, tolerate some FP):")
    logger.info(f"    → Use threshold = {f2_result['threshold']:.4f} (max-F2)")
    logger.info(f"      Expected: ~{f2_result['recall'] * 100:.0f}% events caught, "
                f"~{f2_result['precision'] * 100:.0f}% precision")
    logger.info(f"")
    if 0.7 in recall_results and recall_results[0.7]['achieved']:
        t70 = recall_results[0.7]['threshold']
        logger.info(f"  For BALANCED (≥70% recall with reasonable precision):")
        logger.info(f"    → Use threshold = {t70:.4f}")
        logger.info(f"      Expected: ~{recall_results[0.7]['recall'] * 100:.0f}% recall, "
                    f"~{recall_results[0.7]['precision'] * 100:.0f}% precision")
        logger.info(f"")
    logger.info(f"  For TREND ANALYSIS (best period count accuracy):")
    logger.info(f"    → Use threshold = {period_result['optimal_threshold']:.4f}")
    logger.info(f"      Expected period MAE: {period_result['optimal_mae']:.1f} events")
    logger.info(f"{'=' * 70}")

    # ─── Step 8: Save calibration.json ───
    output_path = args.output or os.path.join(cfg.work_dir, "calibration.json")

    calibration_config = {
        "method": "operating_curve",
        "tiou_match": args.tiou_match,
        "nms_iou_threshold": args.nms_iou,
        "validation_stats": {
            "n_clips": len(result_dict),
            "n_detections": total_preds,
            "n_gt_events": n_gt_val,
            "n_tp": int(label_results[primary_tiou]['is_tp'].sum()),
            "tp_score_mean": float(tp_scores.mean()) if len(tp_scores) > 0 else None,
            "tp_score_p10": float(np.percentile(tp_scores, 10)) if len(tp_scores) >= 5 else None,
            "fp_score_mean": float(fp_scores.mean()) if len(fp_scores) > 0 else None,
        },
        "operating_points": {
            "max_f2": {
                "threshold": f2_result['threshold'],
                "recall": f2_result['recall'],
                "precision": f2_result['precision'],
                "f2": f2_result['f2'],
                "description": "Maximises F2 (recall-weighted). Best for screening.",
            },
            "count_mae_optimal": {
                "threshold": period_result['optimal_threshold'],
                "mae": period_result['optimal_mae'],
                "description": "Minimises period-level count MAE. Best for trend analysis.",
            },
            "target_recall": {
                str(int(k * 100)): {
                    "threshold": v['threshold'],
                    "recall": v['recall'],
                    "precision": v['precision'],
                    "achieved": v['achieved'],
                }
                for k, v in recall_results.items()
            },
        },
        # Default threshold for test.py to use
        "default_threshold": f2_result['threshold'],
        "default_operating_point": "max_f2",
    }

    if isotonic_info is not None:
        calibration_config["isotonic_calibration"] = isotonic_info

    # Include the full operating curve data for plotting
    primary_curve_data = curves[primary_tiou]
    # Subsample for storage
    n_points = min(500, len(primary_curve_data['thresholds']))
    indices = np.linspace(0, len(primary_curve_data['thresholds']) - 1, n_points, dtype=int)
    calibration_config["operating_curve"] = {
        "tiou_match": primary_tiou,
        "thresholds": primary_curve_data['thresholds'][indices].tolist(),
        "precision": primary_curve_data['precision'][indices].tolist(),
        "recall": primary_curve_data['recall'][indices].tolist(),
        "f1": primary_curve_data['f1'][indices].tolist(),
        "f2": primary_curve_data['f2'][indices].tolist(),
    }

    with open(output_path, "w") as f:
        json.dump(calibration_config, f, indent=2)

    logger.info(f"\nCalibration saved to: {output_path}")
    logger.info(f"\nTo use at test time:")
    logger.info(f"  torchrun ... tools/test.py {args.config} \\")
    logger.info(f"    --checkpoint {args.checkpoint} \\")
    logger.info(f"    --calibration {output_path}")
    logger.info(f"\n  Or specify a different operating point:")
    logger.info(f"    --calibration {output_path} --operating_point count_mae_optimal")


if __name__ == "__main__":
    main()