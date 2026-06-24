"""
Test a Temporal Action Detector, optionally with calibrated thresholding.

When --calibration is provided, applies the learned threshold (from the
operating curve) and reports calibrated metrics. A threshold is applied
for the ethologist-oriented reporting. mAP is computed on all predictions
regardless of threshold (since mAP is rank-based).
"""

import os
import sys
import json
import copy

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import numpy as np
import torch
import torch.distributed as dist
import tqdm
from torch.nn.parallel import DistributedDataParallel
from mmengine.config import Config, DictAction

from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader
from opentad.cores import eval_one_epoch
from opentad.utils import update_workdir, set_seed, create_folder, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Test a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--checkpoint", type=str, default="none", help="checkpoint path")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument("--not_eval", action="store_true", help="only do inference")
    parser.add_argument(
        "--calibration", type=str, default=None,
        help="Path to calibration.json from tools/calibrate.py"
    )
    parser.add_argument(
        "--operating_point", type=str, nargs="+", default=None,
        help=(
            "One or more operating points to report: "
            "'max_f2', 'count_mae_optimal', 'recall_70', 'recall_80', etc. "
            "Example: --operating_point max_f2 count_mae_optimal recall_80"
        )
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override: use this raw score threshold directly (ignores calibration file)"
    )
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
    args = parser.parse_args()
    return args


def get_threshold_from_calibration(calibration_config, operating_point=None):
    """Extract the threshold from calibration config for a given operating point."""
    if operating_point is None:
        operating_point = calibration_config.get("default_operating_point", "max_f2")

    ops = calibration_config["operating_points"]

    if operating_point == "max_f2":
        return ops["max_f2"]["threshold"], ops["max_f2"]
    elif operating_point == "count_mae_optimal":
        return ops["count_mae_optimal"]["threshold"], ops["count_mae_optimal"]
    elif operating_point.startswith("recall_"):
        level = operating_point.replace("recall_", "")
        if level in ops["target_recall"]:
            info = ops["target_recall"][level]
            return info["threshold"], info
    # Fallback
    return calibration_config.get("default_threshold", 0.3), {}


def report_calibrated_metrics(result_dict, threshold, cfg, calibration_config, logger):
    """
    At the calibrated threshold, report ethologist-oriented metrics:
    - How many detections survive
    - Clip-level: EDR, count MAE, count recall, specificity, FAR
    - Period-level counts and MAE
    - tIoU profile of accepted detections
    """
    from opentad.calibration.threshold_selection import compute_period_counts_at_threshold

    annotation_path = cfg.evaluation.ground_truth_filename
    subset = cfg.evaluation.subset
    nms_iou = calibration_config.get("nms_iou_threshold", 0.3)

    # Load annotations
    with open(annotation_path, "r") as f:
        ann_data = json.load(f).get("database", {})

    # Count detections above threshold
    n_total = sum(len(v) for v in result_dict.values())
    n_above = sum(1 for preds in result_dict.values() for p in preds if p["score"] >= threshold)

    logger.info(f"\n{'=' * 70}")
    logger.info("CALIBRATED THRESHOLD RESULTS")
    logger.info(f"{'=' * 70}")
    logger.info(f"  Operating point: {calibration_config.get('default_operating_point', 'max_f2')}")
    logger.info(f"  Raw score threshold: {threshold:.4f}")
    logger.info(f"  Total detections: {n_total}")
    logger.info(f"  Above threshold: {n_above} ({n_above / max(n_total, 1) * 100:.1f}%)")

    # ------------------------------------------------------------------
    # Clip-level ethologist metrics at this threshold
    # ------------------------------------------------------------------
    positive_clips = []
    negative_clips = []

    for clip_name, clip_info in ann_data.items():
        if subset is not None and clip_info.get("subset", "") != subset:
            continue
        n_gt = len(clip_info.get("annotations", []))
        if n_gt > 0:
            positive_clips.append(clip_name)
        else:
            negative_clips.append(clip_name)

    logger.info(f"\n  Clip-level ethologist metrics:")
    logger.info(f"    Clips with events: {len(positive_clips)}  |  Clips without events: {len(negative_clips)}")

    # EDR, Count MAE, Count Recall (positive clips)
    tiou_levels_edr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    total_gt_events = 0
    matched_at_tiou = {t: 0 for t in tiou_levels_edr}
    count_errors = []
    count_recalls_list = []

    for clip_name in positive_clips:
        clip_info = ann_data[clip_name]
        gt_segs = np.array(
            [[float(a["segment"][0]), float(a["segment"][1])]
             for a in clip_info.get("annotations", [])],
            dtype=np.float64,
        ).reshape(-1, 2)
        gt_count = len(gt_segs)
        total_gt_events += gt_count

        # Get predictions above threshold for this clip
        predictions = result_dict.get(clip_name, [])
        accepted = [p for p in predictions if p["score"] >= threshold]
        pred_count = len(accepted)

        # Count metrics
        count_errors.append(abs(pred_count - gt_count))
        count_recalls_list.append(min(pred_count, gt_count) / gt_count)

        # EDR: for each GT, check if any accepted prediction matches
        if len(accepted) == 0:
            continue

        pred_segs = np.array(
            [[p["segment"][0], p["segment"][1]] for p in accepted],
            dtype=np.float64,
        )
        for gt_seg in gt_segs:
            tt1 = np.maximum(gt_seg[0], pred_segs[:, 0])
            tt2 = np.minimum(gt_seg[1], pred_segs[:, 1])
            inter = np.clip(tt2 - tt1, 0, None)
            union = (
                (pred_segs[:, 1] - pred_segs[:, 0])
                + (gt_seg[1] - gt_seg[0])
                - inter
            )
            ious = inter / np.clip(union, 1e-8, None)
            best_iou = float(ious.max())
            for tiou_t in tiou_levels_edr:
                if best_iou >= tiou_t:
                    matched_at_tiou[tiou_t] += 1

    # Specificity and FAR (negative clips)
    correct_negatives = 0
    total_false_alarms = 0

    for clip_name in negative_clips:
        predictions = result_dict.get(clip_name, [])
        pred_count = sum(1 for p in predictions if p["score"] >= threshold)
        if pred_count == 0:
            correct_negatives += 1
        total_false_alarms += pred_count

    # Report EDR
    logger.info(f"\n  Event Detection Recall (did we catch real events?):")
    for tiou_t in tiou_levels_edr:
        edr = matched_at_tiou[tiou_t] / max(total_gt_events, 1)
        logger.info(f"    tIoU >= {tiou_t:.2f}: {edr * 100:>5.1f}%")

    # Report Count metrics
    clip_count_mae = float(np.mean(count_errors)) if count_errors else 0.0
    clip_count_recall = float(np.mean(count_recalls_list)) if count_recalls_list else 0.0
    logger.info(f"  Count Metrics (positive-event clips only):")
    logger.info(f"    Count MAE:    {clip_count_mae:.2f} events (avg error in predicted count)")
    logger.info(f"    Count Recall: {clip_count_recall * 100:.1f}% (did we predict enough events?)")

    # Report Specificity and FAR
    logger.info(f"  Empty Clip Behavior:")
    if len(negative_clips) > 0:
        specificity = correct_negatives / len(negative_clips)
        far = total_false_alarms / len(negative_clips)
        logger.info(f"    Specificity:      {specificity * 100:.1f}% (empty clips correctly silent)")
        logger.info(f"    False Alarm Rate: {far:.2f} predictions/empty clip")
    else:
        logger.info(f"    No empty clips in subset.")

    # ------------------------------------------------------------------
    # Period-level counts (existing logic)
    # ------------------------------------------------------------------
    logger.info(f"\n  Period-level counts (NMS tIoU >= {nms_iou}):")
    period_counts = compute_period_counts_at_threshold(
        result_dict, annotation_path, subset=subset,
        threshold=threshold, nms_iou_threshold=nms_iou,
    )

    logger.info(f"    {'Period':<42} {'GT':>4} {'Pred':>4} {'Err':>4}")
    logger.info(f"    {'-' * 56}")
    for period in period_counts['periods']:
        gt_c = period_counts['period_gt'][period]
        pr_c = period_counts['period_pred'][period]
        err = abs(gt_c - pr_c)
        logger.info(f"    {period:<42} {gt_c:>4} {pr_c:>4} {err:>4}")
    logger.info(f"    {'-' * 56}")
    logger.info(f"    {'TOTAL':<42} {period_counts['total_gt']:>4} {period_counts['total_pred']:>4}")
    logger.info(f"    Period MAE: {period_counts['mae']:.2f}")

    # ------------------------------------------------------------------
    # tIoU profile: precision per accepted detection (existing logic)
    # ------------------------------------------------------------------
    logger.info(f"\n  Detection quality profile at threshold={threshold:.4f}:")

    total_gt_profile = 0
    matched_at_tiou_profile = {0.3: 0, 0.5: 0, 0.7: 0}
    total_accepted_profile = 0
    tp_at_tiou = {0.3: 0, 0.5: 0, 0.7: 0}

    for clip_name, predictions in result_dict.items():
        if clip_name not in ann_data:
            continue
        clip_info = ann_data[clip_name]
        if subset is not None and clip_info.get("subset", "") != subset:
            continue

        gt_segs = np.array(
            [[float(a["segment"][0]), float(a["segment"][1])]
             for a in clip_info.get("annotations", [])],
            dtype=np.float64,
        ).reshape(-1, 2)
        total_gt_profile += len(gt_segs)

        accepted = [p for p in predictions if p["score"] >= threshold]
        total_accepted_profile += len(accepted)

        if len(gt_segs) == 0 or len(accepted) == 0:
            continue

        for gt_seg in gt_segs:
            best_iou = 0.0
            for pred in accepted:
                pred_seg = np.array([pred["segment"][0], pred["segment"][1]])
                tt1 = max(gt_seg[0], pred_seg[0])
                tt2 = min(gt_seg[1], pred_seg[1])
                inter = max(0, tt2 - tt1)
                union = (gt_seg[1] - gt_seg[0]) + (pred_seg[1] - pred_seg[0]) - inter
                iou = inter / max(union, 1e-8)
                best_iou = max(best_iou, iou)
            for tiou_level in matched_at_tiou_profile:
                if best_iou >= tiou_level:
                    matched_at_tiou_profile[tiou_level] += 1

        for pred in accepted:
            pred_seg = np.array([pred["segment"][0], pred["segment"][1]])
            best_iou = 0.0
            for gt_seg in gt_segs:
                tt1 = max(gt_seg[0], pred_seg[0])
                tt2 = min(gt_seg[1], pred_seg[1])
                inter = max(0, tt2 - tt1)
                union = (gt_seg[1] - gt_seg[0]) + (pred_seg[1] - pred_seg[0]) - inter
                iou = inter / max(union, 1e-8)
                best_iou = max(best_iou, iou)
            for tiou_level in tp_at_tiou:
                if best_iou >= tiou_level:
                    tp_at_tiou[tiou_level] += 1

    logger.info(f"    Total GT events: {total_gt_profile}")
    logger.info(f"    Accepted detections: {total_accepted_profile}")
    logger.info(f"")
    logger.info(f"    {'tIoU':<8} {'GT Recall':>10} {'Precision':>10}")
    logger.info(f"    {'-' * 32}")
    for tiou_level in sorted(matched_at_tiou_profile.keys()):
        recall = matched_at_tiou_profile[tiou_level] / max(total_gt_profile, 1)
        precision = tp_at_tiou[tiou_level] / max(total_accepted_profile, 1)
        logger.info(f"    >= {tiou_level:<5.1f} {recall * 100:>8.1f}%  {precision * 100:>8.1f}%")

    logger.info(f"{'=' * 70}")


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # DDP init
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["RANK"])
    print(f"Distributed init (rank {args.rank}/{args.world_size}, local rank {args.local_rank})")
    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)

    set_seed(args.seed)
    cfg = update_workdir(cfg, args.id, torch.cuda.device_count())
    if args.rank == 0:
        create_folder(cfg.work_dir)

    logger = setup_logger("Test", save_dir=cfg.work_dir, distributed_rank=args.rank)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")

    # Load calibration if provided
    calibration_config = None
    calibrated_threshold = None
    calibrated_thresholds = []

    if args.calibration is not None:
        with open(args.calibration, "r") as f:
            calibration_config = json.load(f)
        logger.info(f"Loaded calibration from: {args.calibration}")

    if args.threshold is not None:
        # Manual override: single threshold, used as-is
        calibrated_thresholds = [(args.threshold, "manual", {})]
        calibrated_threshold = args.threshold
        logger.info(f"Using manual threshold override: {calibrated_threshold}")
    elif calibration_config is not None:
        # Resolve each requested operating point
        ops_requested = args.operating_point or [
            calibration_config.get("default_operating_point", "max_f2")
        ]
        for op in ops_requested:
            thresh, op_info = get_threshold_from_calibration(calibration_config, op)
            calibrated_thresholds.append((thresh, op, op_info))
            logger.info(f"Operating point '{op}': threshold = {thresh:.4f}")
        # Primary threshold = first in list (used for any single-threshold logic)
        calibrated_threshold = calibrated_thresholds[0][0] if calibrated_thresholds else None

    # Build dataset
    test_dataset = build_dataset(cfg.dataset.test, default_args=dict(logger=logger))
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
    logger.info(f"Using DDP with total {args.world_size} GPUS...")

    if cfg.inference.load_from_raw_predictions:
        logger.info(f"Loading from raw predictions: {cfg.inference.fuse_list}")
    else:
        if args.checkpoint != "none":
            checkpoint_path = args.checkpoint
        elif "test_epoch" in cfg.inference.keys():
            checkpoint_path = os.path.join(
                cfg.work_dir, f"checkpoint/epoch_{cfg.inference.test_epoch}.pth"
            )
        else:
            checkpoint_path = os.path.join(cfg.work_dir, "checkpoint/best.pth")

        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        device = f"cuda:{args.rank % torch.cuda.device_count()}"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        logger.info(f"Checkpoint is epoch {checkpoint['epoch']}.")

        use_ema = getattr(cfg.solver, "ema", False)
        if use_ema:
            model.load_state_dict(checkpoint["state_dict_ema"])
            logger.info("Using Model EMA...")
        else:
            model.load_state_dict(checkpoint["state_dict"])

    use_amp = getattr(cfg.solver, "amp", False)
    if use_amp:
        logger.info("Using Automatic Mixed Precision...")

    # ─── Run standard evaluation (mAP on ALL predictions, unthresholded) ───
    logger.info("Testing Starts...\n")

    # If we have calibration, we need the raw result_dict for additional reporting
    if calibrated_threshold is not None:
        # Run inference manually to get result_dict
        from opentad.cores.test_engine import gather_ddp_results
        from opentad.models.utils.post_processing import build_classifier
        from opentad.datasets.base import SlidingWindowDataset
        from opentad.evaluations import build_evaluator
        from opentad.evaluations.period_counts import compute_period_event_counts

        cfg.inference["folder"] = os.path.join(cfg.work_dir, "outputs")
        if cfg.inference.save_raw_prediction:
            create_folder(cfg.inference["folder"])

        external_cls = None
        if "external_cls" in cfg.post_processing:
            ext_cls_cfg = cfg.post_processing.external_cls
            if ext_cls_cfg is not None:
                if isinstance(ext_cls_cfg, (list, tuple)):
                    external_cls = list(ext_cls_cfg)
                else:
                    external_cls = build_classifier(ext_cls_cfg)

        cfg.post_processing.sliding_window = isinstance(
            test_loader.dataset, SlidingWindowDataset
        )

        model.eval()
        result_dict = {}
        for data_dict in tqdm.tqdm(test_loader, disable=(args.rank != 0)):
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

        if args.rank == 0:
            # Standard mAP evaluation (on ALL predictions — mAP is rank-based)
            result_eval = dict(results=result_dict)

            if not args.not_eval:
                eval_cfg = dict(prediction_filename=result_eval, **cfg.evaluation)
                _compute_period = eval_cfg.pop("compute_period_counts", False)
                _period_score_thresholds = eval_cfg.pop(
                    "period_count_score_thresholds", [0.1, 0.2, 0.3, 0.4, 0.5]
                )
                _period_nms_iou = eval_cfg.pop("period_nms_iou_threshold", 0.3)

                evaluator = build_evaluator(eval_cfg)
                logger.info("Standard evaluation (all predictions, rank-based mAP)...")
                evaluator.evaluate()
                evaluator.logging(logger)

                # Period counts at standard thresholds
                if _compute_period:
                    compute_period_event_counts(
                        result_dict=result_dict,
                        annotation_path=cfg.evaluation.ground_truth_filename,
                        score_thresholds=_period_score_thresholds,
                        subset=cfg.evaluation.subset,
                        nms_iou_threshold=_period_nms_iou,
                        logger=logger,
                        work_dir=cfg.work_dir,
                        named_thresholds=[  
                            (thresh, op_name)
                            for thresh, op_name, _ in calibrated_thresholds
                        ],
                    )

            # ─── Calibrated threshold reports (one per operating point) ───
            for thresh, op_name, op_info in calibrated_thresholds:
                op_cfg = calibration_config if calibration_config is not None else {
                    "nms_iou_threshold": 0.3,
                    "default_operating_point": op_name,
                }
                # Stamp the active operating point into the config so the
                # report header reflects which OP is being shown
                op_cfg = {**op_cfg, "default_operating_point": op_name}
                report_calibrated_metrics(
                    result_dict, thresh, cfg, op_cfg, logger
                )

        logger.info("Testing Over...\n")
    else:
        # No calibration — standard eval_one_epoch
        eval_one_epoch(
            test_loader,
            model,
            cfg,
            logger,
            args.rank,
            model_ema=None,
            use_amp=use_amp,
            world_size=args.world_size,
            not_eval=args.not_eval,
        )
        logger.info("Testing Over...\n")


if __name__ == "__main__":
    main()