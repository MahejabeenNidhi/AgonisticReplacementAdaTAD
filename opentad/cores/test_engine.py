import os
import copy
import json
import tqdm
import torch
import torch.distributed as dist

from opentad.utils import create_folder
from opentad.models.utils.post_processing import build_classifier, batched_nms
from opentad.models.utils.single_class_ext_cls import SingleClassExtCls
from opentad.evaluations import build_evaluator
from opentad.evaluations.period_counts import compute_period_event_counts
from opentad.datasets.base import SlidingWindowDataset


def eval_one_epoch(
    test_loader,
    model,
    cfg,
    logger,
    rank,
    model_ema=None,
    use_amp=False,
    world_size=0,
    not_eval=False,
):
    """Inference and Evaluation the model"""
    # load the ema dict for evaluation
    if model_ema != None:
        current_dict = copy.deepcopy(model.state_dict())
        model.load_state_dict(model_ema.module.state_dict())

    cfg.inference["folder"] = os.path.join(cfg.work_dir, "outputs")
    if cfg.inference.save_raw_prediction:
        create_folder(cfg.inference["folder"])

    # external classifier modified to handle single class better
    external_cls = None
    if "external_cls" in cfg.post_processing:
        ext_cls_cfg = cfg.post_processing.external_cls
        if ext_cls_cfg is not None:
            if isinstance(ext_cls_cfg, (list, tuple)):
                external_cls = list(ext_cls_cfg)
            else:
                external_cls = build_classifier(ext_cls_cfg)

    # whether the testing dataset is sliding window
    cfg.post_processing.sliding_window = isinstance(test_loader.dataset, SlidingWindowDataset)

    # model forward
    model.eval()
    result_dict = {}
    for data_dict in tqdm.tqdm(test_loader, disable=(rank != 0)):
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                results = model(
                    **data_dict,
                    return_loss=False,
                    infer_cfg=cfg.inference,
                    post_cfg=cfg.post_processing,
                    ext_cls=external_cls,
                )
        # update the result dict
        for k, v in results.items():
            if k in result_dict.keys():
                result_dict[k].extend(v)
            else:
                result_dict[k] = v

    result_dict = gather_ddp_results(world_size, result_dict, cfg.post_processing)

    # load back the normal model dict
    if model_ema != None:
        model.load_state_dict(current_dict)

    if rank == 0:
        result_eval = dict(results=result_dict)
        if cfg.post_processing.save_dict:
            result_path = os.path.join(cfg.work_dir, "result_detection.json")
            with open(result_path, "w") as out:
                json.dump(result_eval, out)

        if not not_eval:
            # Build evaluator config
            eval_cfg = dict(prediction_filename=result_eval, **cfg.evaluation)
            # Pop keys used only for period-level counts (not consumed by mAP)
            _compute_period = eval_cfg.pop("compute_period_counts", False)
            _period_score_thresholds = eval_cfg.pop("period_count_score_thresholds", [0.1, 0.2, 0.3, 0.4, 0.5])
            _period_nms_iou = eval_cfg.pop("period_nms_iou_threshold", 0.3)

            evaluator = build_evaluator(eval_cfg)
            # evaluate and output
            logger.info("Evaluation starts...")
            metrics_dict = evaluator.evaluate()
            evaluator.logging(logger)

            # --- Period event counts (cross-clip dedup) ---
            if _compute_period:
                ann_path = cfg.evaluation.ground_truth_filename
                eval_subset = cfg.evaluation.subset
                compute_period_event_counts(
                    result_dict=result_dict,
                    annotation_path=ann_path,
                    score_thresholds=_period_score_thresholds,
                    subset=eval_subset,
                    nms_iou_threshold=_period_nms_iou,
                    logger=logger,
                    work_dir=cfg.work_dir,
                )

            return metrics_dict

    return None


def gather_ddp_results(world_size, result_dict, post_cfg):
    gather_dict_list = [None for _ in range(world_size)]
    dist.all_gather_object(gather_dict_list, result_dict)
    result_dict = {}
    for i in range(world_size):  # update the result dict
        for k, v in gather_dict_list[i].items():
            if k in result_dict.keys():
                result_dict[k].extend(v)
            else:
                result_dict[k] = v

    # do nms for sliding window, if needed
    if post_cfg.sliding_window == True and post_cfg.nms is not None:
        # assert sliding_window=True
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
                # convert to python scalars
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
