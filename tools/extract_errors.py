#!/usr/bin/env python3
"""
Extract video segments for false alarms and missed detections at a
calibrated operating point (e.g., max_f2).

This enables qualitative analysis of model errors:
- False alarms: predicted events above threshold with no matching GT
- Missed detections: GT events not matched by any prediction above threshold

Usage:
    torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
        tools/extract_errors.py \
        configs/adatad/displacement/e2e_displacement_2min_videomaev2_b_160x2_160_adapter_predecoded.py \
        --checkpoint exps/displacement/adatad/.../checkpoint/best.pth \
        --calibration exps/displacement/adatad/.../calibration.json \
        --operating_point max_f2 \
        --output_dir qualitative_analysis/ratio_1.0_maxf2 \
        --num_false_alarms 50 \
        --num_missed 50 \
        --tiou_match 0.3 \
        --context_pad 3.0 \
        --seed 42

Outputs:
    output_dir/
        false_alarm/
            FA_001_{clip}_{start:.1f}-{end:.1f}_s{score:.3f}.mp4
            ...
        missed_detection/
            MD_001_{clip}_{start:.1f}-{end:.1f}.mp4
            ...
        error_analysis_summary.json
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
import tqdm

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from mmengine.config import Config, DictAction

from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader
from opentad.cores.test_engine import gather_ddp_results
from opentad.models.utils.post_processing import build_classifier
from opentad.datasets.base import SlidingWindowDataset
from opentad.utils import update_workdir, set_seed, create_folder, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract false alarm and missed detection video segments"
    )
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint path")
    parser.add_argument(
        "--calibration", type=str, required=True,
        help="Path to calibration.json (from tools/calibrate.py)"
    )
    parser.add_argument(
        "--operating_point", type=str, default="max_f2",
        help="Operating point to use: 'max_f2', 'count_mae_optimal', 'recall_70', etc."
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override: use this raw score threshold directly (ignores calibration)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for extracted clips"
    )
    parser.add_argument(
        "--num_false_alarms", type=int, default=50,
        help="Number of false alarm clips to extract (randomly sampled)"
    )
    parser.add_argument(
        "--num_missed", type=int, default=50,
        help="Number of missed detection clips to extract (randomly sampled)"
    )
    parser.add_argument(
        "--tiou_match", type=float, default=0.3,
        help="tIoU threshold for matching predictions to GT"
    )
    parser.add_argument(
        "--context_pad", type=float, default=3.0,
        help="Seconds of context padding before/after the event in extracted clips"
    )
    parser.add_argument(
        "--video_dir", type=str, default=None,
        help="Directory containing source videos. If None, uses data_path from config."
    )
    parser.add_argument(
        "--subset", type=str, default=None,
        help="Subset to analyse. If None, uses evaluation.subset from config."
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for sampling")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument(
        "--predictions_json", type=str, default=None,
        help="Load predictions from a saved JSON instead of running inference"
    )
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
    args = parser.parse_args()
    return args


def get_threshold_from_calibration(calibration_config, operating_point):
    """Extract the threshold for a given operating point."""
    ops = calibration_config["operating_points"]
    if operating_point == "max_f2":
        return ops["max_f2"]["threshold"], ops["max_f2"]
    elif operating_point == "count_mae_optimal":
        return ops["count_mae_optimal"]["threshold"], ops["count_mae_optimal"]
    elif operating_point.startswith("recall_"):
        level = operating_point.replace("recall_", "")
        if level in ops.get("target_recall", {}):
            info = ops["target_recall"][level]
            return info["threshold"], info
    # Fallback
    return calibration_config.get("default_threshold", 0.3), {}


def segment_iou(target, candidates):
    """Compute tIoU between one target segment [s, e] and N candidate segments."""
    candidates = np.asarray(candidates, dtype=np.float64).reshape(-1, 2)
    tt1 = np.maximum(target[0], candidates[:, 0])
    tt2 = np.minimum(target[1], candidates[:, 1])
    inter = np.clip(tt2 - tt1, 0, None)
    union = (candidates[:, 1] - candidates[:, 0]) + (target[1] - target[0]) - inter
    return inter / np.clip(union, 1e-8, None)


def identify_errors(result_dict, ann_data, subset, threshold, tiou_match):
    """
    Identify false alarms and missed detections across all clips.

    Returns
    -------
    false_alarms : list of dict
        Each entry: {clip_name, segment, score, label, source_video, source_start, duration}
    missed_detections : list of dict
        Each entry: {clip_name, segment, source_video, source_start, duration,
                     best_pred_score (if any partial match), best_pred_tiou}
    """
    false_alarms = []
    missed_detections = []

    for clip_name, clip_info in ann_data.items():
        if subset is not None and clip_info.get("subset", "") != subset:
            continue

        # GT segments (clip-local seconds)
        gt_segs = np.array(
            [[float(a["segment"][0]), float(a["segment"][1])]
             for a in clip_info.get("annotations", [])],
            dtype=np.float64,
        ).reshape(-1, 2)
        n_gt = len(gt_segs)

        # Source video info for extraction
        source_video = clip_info.get("source_video", clip_name + ".mp4")
        source_start = clip_info.get("source_start", 0.0)
        duration = clip_info.get("duration", 120.0)

        # Predictions for this clip above threshold
        predictions = result_dict.get(clip_name, [])
        preds_above = [p for p in predictions if p["score"] >= threshold]
        preds_above = sorted(preds_above, key=lambda x: x["score"], reverse=True)

        n_pred = len(preds_above)

        if n_gt == 0 and n_pred == 0:
            continue

        # Greedy matching: highest-scoring prediction first
        gt_matched = np.zeros(n_gt, dtype=bool)
        pred_is_tp = np.zeros(n_pred, dtype=bool)

        for pred_idx, pred in enumerate(preds_above):
            if n_gt == 0:
                break
            pred_seg = np.array([pred["segment"][0], pred["segment"][1]])
            tiou_arr = segment_iou(pred_seg, gt_segs)

            # Find best unmatched GT
            sorted_gt_idx = np.argsort(-tiou_arr)
            for gt_idx in sorted_gt_idx:
                if tiou_arr[gt_idx] < tiou_match:
                    break
                if not gt_matched[gt_idx]:
                    pred_is_tp[pred_idx] = True
                    gt_matched[gt_idx] = True
                    break

        # Collect false alarms (unmatched predictions)
        for pred_idx, pred in enumerate(preds_above):
            if not pred_is_tp[pred_idx]:
                # Find the best tIoU with any GT (for context)
                best_tiou_with_gt = 0.0
                if n_gt > 0:
                    pred_seg = np.array([pred["segment"][0], pred["segment"][1]])
                    tiou_arr = segment_iou(pred_seg, gt_segs)
                    best_tiou_with_gt = float(tiou_arr.max())

                false_alarms.append(dict(
                    clip_name=clip_name,
                    segment=[pred["segment"][0], pred["segment"][1]],
                    score=pred["score"],
                    label=pred.get("label", "displacement"),
                    source_video=source_video,
                    source_start=source_start,
                    duration=duration,
                    best_tiou_with_gt=best_tiou_with_gt,
                ))

        # Collect missed detections (unmatched GT)
        for gt_idx in range(n_gt):
            if not gt_matched[gt_idx]:
                # Find the best prediction that partially overlaps this GT
                best_pred_score = 0.0
                best_pred_tiou = 0.0
                best_pred_below_thresh_score = 0.0

                # Check ALL predictions (including below threshold)
                all_preds_for_clip = sorted(predictions, key=lambda x: x["score"], reverse=True)
                for pred in all_preds_for_clip:
                    pred_seg = np.array([pred["segment"][0], pred["segment"][1]])
                    gt_seg = gt_segs[gt_idx]
                    tiou_val = float(segment_iou(gt_seg, pred_seg.reshape(1, 2))[0])
                    if tiou_val > best_pred_tiou:
                        best_pred_tiou = tiou_val
                        best_pred_score = pred["score"]
                    if pred["score"] < threshold and tiou_val >= tiou_match:
                        best_pred_below_thresh_score = max(
                            best_pred_below_thresh_score, pred["score"]
                        )

                missed_detections.append(dict(
                    clip_name=clip_name,
                    segment=[float(gt_segs[gt_idx][0]), float(gt_segs[gt_idx][1])],
                    source_video=source_video,
                    source_start=source_start,
                    duration=duration,
                    best_pred_score=best_pred_score,
                    best_pred_tiou=best_pred_tiou,
                    best_pred_below_thresh_score=best_pred_below_thresh_score,
                ))

    return false_alarms, missed_detections


def extract_clip_ffmpeg(
    source_video_path,
    output_path,
    start_sec,
    end_sec,
    context_pad=3.0,
):
    """
    Extract a video segment using ffmpeg (copy codec for speed).
    Falls back to re-encoding if copy fails.

    Parameters
    ----------
    source_video_path : str
        Path to the full source video.
    output_path : str
        Where to save the extracted clip.
    start_sec : float
        Absolute start time in the source video.
    end_sec : float
        Absolute end time in the source video.
    context_pad : float
        Seconds of context to add before and after.
    """
    # Add context padding
    padded_start = max(0, start_sec - context_pad)
    padded_end = end_sec + context_pad
    duration = padded_end - padded_start

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Try stream copy first (fast, lossless)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{padded_start:.3f}",
        "-i", source_video_path,
        "-t", f"{duration:.3f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        output_path,
    ]

    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60
    )

    if result.returncode != 0:
        # Fallback: re-encode (slower but more reliable for precise cuts)
        cmd_reencode = [
            "ffmpeg", "-y",
            "-ss", f"{padded_start:.3f}",
            "-i", source_video_path,
            "-t", f"{duration:.3f}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "aac",
            output_path,
        ]
        subprocess.run(
            cmd_reencode, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120
        )


def run_inference(cfg, args, logger):
    """Run model inference and return result_dict."""
    # Build dataset
    test_cfg_override = cfg.dataset.test.copy()
    if args.subset is not None:
        test_cfg_override["subset_name"] = args.subset

    test_dataset = build_dataset(test_cfg_override, default_args=dict(logger=logger))
    test_loader = build_dataloader(
        test_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=False,
        drop_last=False,
        seed=args.seed,
        **cfg.solver.test,
    )

    # Build model
    model = build_detector(cfg.model)
    model = model.to(args.local_rank)
    model = DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank
    )

    # Load checkpoint
    device = f"cuda:{args.local_rank}"
    checkpoint = torch.load(args.checkpoint, map_location=device)
    logger.info(f"Loaded checkpoint epoch {checkpoint['epoch']}")

    use_ema = getattr(cfg.solver, "ema", False)
    if use_ema and "state_dict_ema" in checkpoint:
        model.load_state_dict(checkpoint["state_dict_ema"])
        logger.info("Using Model EMA weights")
    else:
        model.load_state_dict(checkpoint["state_dict"])

    use_amp = getattr(cfg.solver, "amp", False)

    # External classifier
    external_cls = None
    if "external_cls" in cfg.post_processing:
        ext_cls_cfg = cfg.post_processing.external_cls
        if ext_cls_cfg is not None:
            if isinstance(ext_cls_cfg, (list, tuple)):
                external_cls = list(ext_cls_cfg)
            else:
                external_cls = build_classifier(ext_cls_cfg)

    cfg.post_processing.sliding_window = isinstance(test_loader.dataset, SlidingWindowDataset)
    cfg.inference["folder"] = os.path.join(cfg.work_dir, "outputs")

    # Run inference
    model.eval()
    result_dict = {}
    for data_dict in tqdm.tqdm(test_loader, disable=(args.rank != 0), desc="Inference"):
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

    # Setup logger
    logger = setup_logger("ExtractErrors", save_dir=cfg.work_dir, distributed_rank=args.rank)

    # Determine threshold
    if args.threshold is not None:
        threshold = args.threshold
        op_name = "manual"
        op_info = {}
        logger.info(f"Using manual threshold: {threshold:.4f}")
    else:
        with open(args.calibration, "r") as f:
            calibration_config = json.load(f)
        threshold, op_info = get_threshold_from_calibration(
            calibration_config, args.operating_point
        )
        op_name = args.operating_point
        logger.info(f"Operating point '{op_name}': threshold = {threshold:.4f}")
        if op_info:
            logger.info(f"  Expected recall: {op_info.get('recall', 'N/A')}")
            logger.info(f"  Expected precision: {op_info.get('precision', 'N/A')}")

    # Determine subset
    subset = args.subset if args.subset is not None else cfg.evaluation.get("subset", "testing")
    logger.info(f"Analysing subset: {subset}")

    # Get predictions
    if args.predictions_json is not None:
        logger.info(f"Loading predictions from: {args.predictions_json}")
        with open(args.predictions_json, "r") as f:
            pred_data = json.load(f)
        result_dict = pred_data.get("results", pred_data)
    else:
        logger.info("Running inference...")
        result_dict = run_inference(cfg, args, logger)

    # Only rank 0 does the analysis and extraction
    if args.rank != 0:
        logger.info("Non-rank-0 process exiting.")
        return

    # Load annotations
    annotation_path = cfg.evaluation.ground_truth_filename
    with open(annotation_path, "r") as f:
        ann_data = json.load(f).get("database", {})

    # Identify errors
    logger.info(f"\nIdentifying errors at threshold={threshold:.4f}, tIoU_match={args.tiou_match}...")
    false_alarms, missed_detections = identify_errors(
        result_dict, ann_data, subset, threshold, args.tiou_match
    )

    logger.info(f"  Total false alarms found: {len(false_alarms)}")
    logger.info(f"  Total missed detections found: {len(missed_detections)}")

    # Count total GT and predictions above threshold for context
    total_gt = sum(
        len(info.get("annotations", []))
        for info in ann_data.values()
        if subset is None or info.get("subset", "") == subset
    )
    total_pred_above = sum(
        1 for preds in result_dict.values()
        for p in preds if p["score"] >= threshold
    )
    n_tp = total_pred_above - len(false_alarms)
    logger.info(f"  Total GT events in {subset}: {total_gt}")
    logger.info(f"  Total predictions above threshold: {total_pred_above}")
    logger.info(f"  True positives: {n_tp}")
    logger.info(f"  Recall: {n_tp / max(total_gt, 1) * 100:.1f}%")
    logger.info(f"  Precision: {n_tp / max(total_pred_above, 1) * 100:.1f}%")

    # Sample errors
    rng = np.random.RandomState(args.seed)

    n_fa_sample = min(args.num_false_alarms, len(false_alarms))
    n_md_sample = min(args.num_missed, len(missed_detections))

    if n_fa_sample < args.num_false_alarms:
        logger.info(
            f"  NOTE: Only {len(false_alarms)} false alarms available "
            f"(requested {args.num_false_alarms})"
        )
    if n_md_sample < args.num_missed:
        logger.info(
            f"  NOTE: Only {len(missed_detections)} missed detections available "
            f"(requested {args.num_missed})"
        )

    # Sort false alarms by score (descending) before sampling to get a
    # mix of high-confidence and lower-confidence errors
    false_alarms_sorted = sorted(false_alarms, key=lambda x: x["score"], reverse=True)
    missed_sorted = sorted(missed_detections, key=lambda x: x["best_pred_score"], reverse=True)

    if n_fa_sample > 0:
        fa_indices = rng.choice(len(false_alarms_sorted), size=n_fa_sample, replace=False)
        sampled_fa = [false_alarms_sorted[i] for i in sorted(fa_indices)]
    else:
        sampled_fa = []

    if n_md_sample > 0:
        md_indices = rng.choice(len(missed_sorted), size=n_md_sample, replace=False)
        sampled_md = [missed_sorted[i] for i in sorted(md_indices)]
    else:
        sampled_md = []

    # Determine video directory
    video_dir = args.video_dir
    if video_dir is None:
        # Try to get from config
        if isinstance(cfg.dataset.test.data_path, str):
            video_dir = cfg.dataset.test.data_path
        else:
            video_dir = cfg.dataset.test.data_path[0]
    logger.info(f"\nVideo source directory: {video_dir}")

    # Create output directories
    fa_dir = os.path.join(args.output_dir, "false_alarm")
    md_dir = os.path.join(args.output_dir, "missed_detection")
    os.makedirs(fa_dir, exist_ok=True)
    os.makedirs(md_dir, exist_ok=True)

    # Extract false alarm clips
    logger.info(f"\nExtracting {n_fa_sample} false alarm clips...")
    fa_metadata = []
    for i, fa in enumerate(tqdm.tqdm(sampled_fa, desc="False alarms")):
        clip_start = fa["segment"][0]
        clip_end = fa["segment"][1]

        # Absolute time in source video
        abs_start = fa["source_start"] + clip_start
        abs_end = fa["source_start"] + clip_end

        # Build output filename
        safe_clip = fa["clip_name"].replace("/", "_")
        filename = (
            f"FA_{i + 1:03d}_{safe_clip}_"
            f"t{clip_start:.1f}-{clip_end:.1f}_"
            f"s{fa['score']:.3f}.mp4"
        )
        output_path = os.path.join(fa_dir, filename)

        # Find source video
        source_video_path = os.path.join(video_dir, fa["source_video"])
        if not os.path.exists(source_video_path):
            # Try clip name as video file
            alt_path = os.path.join(video_dir, fa["clip_name"] + ".mp4")
            if os.path.exists(alt_path):
                source_video_path = alt_path
                abs_start = clip_start
                abs_end = clip_end
            else:
                logger.warning(f"  Source video not found: {source_video_path}")
                fa_metadata.append({**fa, "extracted": False, "reason": "video_not_found"})
                continue

        extract_clip_ffmpeg(
            source_video_path, output_path,
            abs_start, abs_end,
            context_pad=args.context_pad,
        )

        fa_metadata.append({
            **fa,
            "extracted": True,
            "output_file": filename,
            "abs_start": abs_start,
            "abs_end": abs_end,
            "context_pad": args.context_pad,
        })

    # Extract missed detection clips
    logger.info(f"\nExtracting {n_md_sample} missed detection clips...")
    md_metadata = []
    for i, md in enumerate(tqdm.tqdm(sampled_md, desc="Missed detections")):
        clip_start = md["segment"][0]
        clip_end = md["segment"][1]

        # Absolute time in source video
        abs_start = md["source_start"] + clip_start
        abs_end = md["source_start"] + clip_end

        # Build output filename
        safe_clip = md["clip_name"].replace("/", "_")
        filename = (
            f"MD_{i + 1:03d}_{safe_clip}_"
            f"t{clip_start:.1f}-{clip_end:.1f}_"
            f"bestScore{md['best_pred_score']:.3f}.mp4"
        )
        output_path = os.path.join(md_dir, filename)

        # Find source video
        source_video_path = os.path.join(video_dir, md["source_video"])
        if not os.path.exists(source_video_path):
            alt_path = os.path.join(video_dir, md["clip_name"] + ".mp4")
            if os.path.exists(alt_path):
                source_video_path = alt_path
                abs_start = clip_start
                abs_end = clip_end
            else:
                logger.warning(f"  Source video not found: {source_video_path}")
                md_metadata.append({**md, "extracted": False, "reason": "video_not_found"})
                continue

        extract_clip_ffmpeg(
            source_video_path, output_path,
            abs_start, abs_end,
            context_pad=args.context_pad,
        )

        md_metadata.append({
            **md,
            "extracted": True,
            "output_file": filename,
            "abs_start": abs_start,
            "abs_end": abs_end,
            "context_pad": args.context_pad,
        })

    # Save summary JSON
    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "calibration": args.calibration,
        "operating_point": op_name,
        "threshold": threshold,
        "tiou_match": args.tiou_match,
        "subset": subset,
        "context_pad_seconds": args.context_pad,
        "seed": args.seed,
        "stats": {
            "total_gt_events": total_gt,
            "total_predictions_above_threshold": total_pred_above,
            "true_positives": n_tp,
            "total_false_alarms": len(false_alarms),
            "total_missed_detections": len(missed_detections),
            "recall": n_tp / max(total_gt, 1),
            "precision": n_tp / max(total_pred_above, 1),
            "sampled_false_alarms": n_fa_sample,
            "sampled_missed_detections": n_md_sample,
        },
        "false_alarm_score_distribution": {
            "min": float(min(fa["score"] for fa in false_alarms)) if false_alarms else 0,
            "max": float(max(fa["score"] for fa in false_alarms)) if false_alarms else 0,
            "mean": float(np.mean([fa["score"] for fa in false_alarms])) if false_alarms else 0,
            "median": float(np.median([fa["score"] for fa in false_alarms])) if false_alarms else 0,
        },
        "missed_detection_analysis": {
            "n_with_partial_overlap": sum(
                1 for md in missed_detections if md["best_pred_tiou"] > 0
            ),
            "n_with_pred_below_threshold": sum(
                1 for md in missed_detections if md["best_pred_below_thresh_score"] > 0
            ),
            "mean_best_pred_tiou": float(np.mean(
                [md["best_pred_tiou"] for md in missed_detections]
            )) if missed_detections else 0,
        },
        "false_alarms": fa_metadata,
        "missed_detections": md_metadata,
    }

    summary_path = os.path.join(args.output_dir, "error_analysis_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print final summary
    logger.info(f"\n{'=' * 70}")
    logger.info("ERROR EXTRACTION COMPLETE")
    logger.info(f"{'=' * 70}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  False alarms extracted: {sum(1 for m in fa_metadata if m.get('extracted', False))}/{n_fa_sample}")
    logger.info(f"  Missed detections extracted: {sum(1 for m in md_metadata if m.get('extracted', False))}/{n_md_sample}")
    logger.info(f"  Summary saved to: {summary_path}")
    logger.info(f"")
    logger.info(f"  False alarm score range: [{summary['false_alarm_score_distribution']['min']:.3f}, "
                f"{summary['false_alarm_score_distribution']['max']:.3f}]")
    logger.info(f"  False alarm mean score: {summary['false_alarm_score_distribution']['mean']:.3f}")
    logger.info(f"")
    logger.info(f"  Missed detections with partial overlap (tIoU > 0): "
                f"{summary['missed_detection_analysis']['n_with_partial_overlap']}/{len(missed_detections)}")
    logger.info(f"  Missed detections with pred below threshold: "
                f"{summary['missed_detection_analysis']['n_with_pred_below_threshold']}/{len(missed_detections)}")
    logger.info(f"  Mean best tIoU of partial matches: "
                f"{summary['missed_detection_analysis']['mean_best_pred_tiou']:.3f}")
    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
