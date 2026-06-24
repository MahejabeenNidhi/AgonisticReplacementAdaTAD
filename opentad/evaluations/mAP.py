import json
import numpy as np
import pandas as pd
import multiprocessing as mp

from .builder import EVALUATORS, remove_duplicate_annotations


@EVALUATORS.register_module()
class mAP:
    def __init__(
        self,
        ground_truth_filename,
        prediction_filename,
        subset,
        tiou_thresholds,
        top_k=None,
        blocked_videos=None,
        thread=16,
        compute_count_metrics=False,
        count_score_thresholds=None,
    ):
        super().__init__()
        if not ground_truth_filename:
            raise IOError("Please input a valid ground truth file.")
        if not prediction_filename:
            raise IOError("Please input a valid prediction file.")
        self.subset = subset
        self.tiou_thresholds = tiou_thresholds
        self.top_k = top_k
        self.gt_fields = ["database"]
        self.pred_fields = ["results"]
        self.thread = thread

        # Count & Recall metric settings
        self.compute_count_metrics = compute_count_metrics
        if count_score_thresholds is None:
            self.count_score_thresholds = [0.3, 0.5]
        else:
            self.count_score_thresholds = count_score_thresholds

        # Get blocked videos
        if blocked_videos is None:
            self.blocked_videos = list()
        else:
            with open(blocked_videos) as json_file:
                self.blocked_videos = json.load(json_file)

        # Import ground truth and predictions.
        self.ground_truth, self.activity_index = self._import_ground_truth(ground_truth_filename)
        self.prediction = self._import_prediction(prediction_filename)

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
        the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        with open(ground_truth_filename, "r") as fobj:
            data = json.load(fobj)

        # Checking format
        if not all([field in list(data.keys()) for field in self.gt_fields]):
            raise IOError("Please input a valid ground truth file.")

        # Read ground truth data.
        activity_index, cidx = {}, 0
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []

        # Track all videos in this subset for count metrics (including 0-event videos)
        self.all_video_ids = []
        self.gt_count_per_video = {}
        self.gt_segments_per_video = {}

        for videoid, v in data["database"].items():
            if self.subset != v["subset"]:
                continue
            if videoid in self.blocked_videos:
                continue

            self.all_video_ids.append(videoid)

            # remove duplicated instances following ActionFormer
            v_anno = remove_duplicate_annotations(v["annotations"])

            # Store GT count and segments per video
            self.gt_count_per_video[videoid] = len(v_anno)
            self.gt_segments_per_video[videoid] = [
                (float(ann["segment"][0]), float(ann["segment"][1])) for ann in v_anno
            ]

            for ann in v_anno:
                if ann["label"] not in activity_index:
                    activity_index[ann["label"]] = cidx
                    cidx += 1
                video_lst.append(videoid)
                t_start_lst.append(float(ann["segment"][0]))
                t_end_lst.append(float(ann["segment"][1]))
                label_lst.append(activity_index[ann["label"]])

        ground_truth = pd.DataFrame(
            {
                "video-id": video_lst,
                "t-start": t_start_lst,
                "t-end": t_end_lst,
                "label": label_lst,
            }
        )
        return ground_truth, activity_index

    def _import_prediction(self, prediction_filename):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.

        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.

        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        # if prediction_filename is a string, then json load
        if isinstance(prediction_filename, str):
            with open(prediction_filename, "r") as fobj:
                data = json.load(fobj)
        elif isinstance(prediction_filename, dict):
            data = prediction_filename
        else:
            raise IOError(f"Type of prediction file is {type(prediction_filename)}.")

        # Checking format...
        if not all([field in list(data.keys()) for field in self.pred_fields]):
            raise IOError("Please input a valid prediction file.")

        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst = [], []
        for video_id, v in data["results"].items():
            if video_id in self.blocked_videos:
                continue
            for result in v:
                try:
                    label = self.activity_index[result["label"]]
                except:
                    # this is because the predicted label is not in annotation
                    # such as the some classes only exists in train split, but not in val split
                    label = len(self.activity_index)
                video_lst.append(video_id)
                t_start_lst.append(float(result["segment"][0]))
                t_end_lst.append(float(result["segment"][1]))
                label_lst.append(label)
                score_lst.append(result["score"])
        prediction = pd.DataFrame(
            {
                "video-id": video_lst,
                "t-start": t_start_lst,
                "t-end": t_end_lst,
                "label": label_lst,
                "score": score_lst,
            }
        )
        return prediction

    def wrapper_compute_average_precision(self, cidx_list):
        """Computes average precision for a sub class list."""
        for cidx in cidx_list:
            gt_idx = self.ground_truth["label"] == cidx
            pred_idx = self.prediction["label"] == cidx
            self.mAP_result_dict[cidx] = compute_average_precision_detection(
                self.ground_truth.loc[gt_idx].reset_index(drop=True),
                self.prediction.loc[pred_idx].reset_index(drop=True),
                tiou_thresholds=self.tiou_thresholds,
            )

    def wrapper_compute_topkx_recall(self, cidx_list):
        """Computes Top-kx recall for a sub class list."""
        for cidx in cidx_list:
            gt_idx = self.ground_truth["label"] == cidx
            pred_idx = self.prediction["label"] == cidx
            self.recall_result_dict[cidx] = compute_topkx_recall_detection(
                self.ground_truth.loc[gt_idx].reset_index(drop=True),
                self.prediction.loc[pred_idx].reset_index(drop=True),
                tiou_thresholds=self.tiou_thresholds,
                top_k=self.top_k,
            )

    def multi_thread_compute_average_precision(self):
        self.mAP_result_dict = mp.Manager().dict()

        num_total = len(self.activity_index.values())
        num_activity_per_thread = num_total // self.thread + 1

        processes = []
        for tid in range(self.thread):
            num_start = int(tid * num_activity_per_thread)
            num_end = min(num_start + num_activity_per_thread, num_total)

            p = mp.Process(
                target=self.wrapper_compute_average_precision,
                args=(list(self.activity_index.values())[num_start:num_end],),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index.items())))
        for i, cidx in enumerate(self.activity_index.values()):
            ap[:, cidx] = self.mAP_result_dict[i]
        return ap

    def multi_thread_compute_topkx_recall(self):
        self.recall_result_dict = mp.Manager().dict()

        num_total = len(self.activity_index.values())
        num_activity_per_thread = num_total // self.thread + 1

        processes = []
        for tid in range(self.thread):
            num_start = int(tid * num_activity_per_thread)
            num_end = min(num_start + num_activity_per_thread, num_total)

            p = mp.Process(
                target=self.wrapper_compute_topkx_recall,
                args=(list(self.activity_index.values())[num_start:num_end],),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        recall = np.zeros((len(self.tiou_thresholds), len(self.top_k), len(self.activity_index.items())))
        for i, cidx in enumerate(self.activity_index.values()):
            recall[..., cidx] = self.recall_result_dict[i]
        return recall

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        self.ap = self.multi_thread_compute_average_precision()
        self.mAPs = self.ap.mean(axis=1)
        self.average_mAP = self.mAPs.mean()

        metric_dict = dict(average_mAP=self.average_mAP)
        for tiou, mAP in zip(self.tiou_thresholds, self.mAPs):
            metric_dict[f"mAP@{tiou}"] = mAP

        # if top_k is not None, we will compute top-kx recall
        if self.top_k is not None:
            self.recall = self.multi_thread_compute_topkx_recall()
            self.mRecall = self.recall.mean(axis=2)
            for tiou, mRecall in zip(self.tiou_thresholds, self.mRecall):
                for k, recall in zip(self.top_k, mRecall):
                    metric_dict[f"recall@{tiou}@{k}"] = recall

        # Compute ethologist-oriented count and recall metrics
        if self.compute_count_metrics:
            self.count_metrics = self._compute_count_and_recall_metrics()
            metric_dict.update(self.count_metrics)

        return metric_dict

    def _compute_count_and_recall_metrics(self):
        """Compute event detection recall, count MAE, count recall, and specificity.

        For videos WITH ground truth events:
          - Event Detection Recall: fraction of GT events that are matched by a
            prediction with tIoU >= threshold. This is the most critical metric
            for an ethologist ("did we catch the real events?").
          - Count MAE: mean |predicted_count - gt_count| across positive videos.
          - Count Recall: mean of min(pred_count, gt_count) / gt_count. Measures
            whether the model predicts "enough" events (bounded [0, 1]).

        For videos WITHOUT ground truth events:
          - Zero-event Specificity: fraction of empty clips with 0 predictions.
          - False Alarm Rate: average number of predictions on empty clips.

        Returns
        -------
        results : dict
        """
        results = {}

        # Separate videos into positive (has events) and negative (no events)
        positive_videos = [v for v in self.all_video_ids if self.gt_count_per_video[v] > 0]
        negative_videos = [v for v in self.all_video_ids if self.gt_count_per_video[v] == 0]
        results["n_positive_videos"] = len(positive_videos)
        results["n_negative_videos"] = len(negative_videos)

        # Pre-group predictions by video-id
        if not self.prediction.empty:
            pred_gbvn = self.prediction.groupby("video-id")
        else:
            pred_gbvn = None

        for score_thresh in self.count_score_thresholds:
            # ─── Event Detection Recall (tIoU-matched) ───
            # For each tIoU threshold, compute: of all GT events, how many were matched?
            for tiou_thresh in self.tiou_thresholds:
                total_gt_events = 0
                matched_gt_events = 0

                for video_id in positive_videos:
                    gt_segs = np.array(self.gt_segments_per_video[video_id])  # (N_gt, 2)
                    total_gt_events += len(gt_segs)

                    # Get predictions for this video above score threshold
                    pred_segs = self._get_pred_segments(pred_gbvn, video_id, score_thresh)

                    if len(pred_segs) == 0:
                        continue

                    # For each GT event, check if ANY prediction matches with tIoU >= threshold
                    for gt_seg in gt_segs:
                        tiou_arr = segment_iou(gt_seg, pred_segs)
                        if tiou_arr.max() >= tiou_thresh:
                            matched_gt_events += 1

                if total_gt_events > 0:
                    event_recall = matched_gt_events / total_gt_events
                else:
                    event_recall = 0.0
                results[f"event_recall@tIoU{tiou_thresh}@s{score_thresh}"] = event_recall

            # ─── Count MAE & Count Recall (positive videos only) ───
            count_errors = []
            count_recalls = []

            for video_id in positive_videos:
                gt_count = self.gt_count_per_video[video_id]
                pred_segs = self._get_pred_segments(pred_gbvn, video_id, score_thresh)
                pred_count = len(pred_segs)

                count_errors.append(abs(pred_count - gt_count))
                # Count recall: did we predict at least as many as actually exist?
                # min(pred, gt) / gt: 1.0 means we predicted >= gt events
                count_recalls.append(min(pred_count, gt_count) / gt_count)

            if len(count_errors) > 0:
                results[f"count_MAE_positive@s{score_thresh}"] = np.mean(count_errors)
                results[f"count_recall@s{score_thresh}"] = np.mean(count_recalls)
            else:
                results[f"count_MAE_positive@s{score_thresh}"] = 0.0
                results[f"count_recall@s{score_thresh}"] = 0.0

            # ─── Zero-event Specificity & False Alarm Rate ───
            if len(negative_videos) > 0:
                correct_negatives = 0
                total_false_alarms = 0

                for video_id in negative_videos:
                    pred_segs = self._get_pred_segments(pred_gbvn, video_id, score_thresh)
                    pred_count = len(pred_segs)
                    if pred_count == 0:
                        correct_negatives += 1
                    total_false_alarms += pred_count

                results[f"zero_event_specificity@s{score_thresh}"] = correct_negatives / len(negative_videos)
                results[f"false_alarm_rate@s{score_thresh}"] = total_false_alarms / len(negative_videos)
            else:
                results[f"zero_event_specificity@s{score_thresh}"] = float("nan")
                results[f"false_alarm_rate@s{score_thresh}"] = float("nan")

        return results

    def _get_pred_segments(self, pred_gbvn, video_id, score_thresh):
        """Get predicted segments for a video filtered by score threshold.

        Returns
        -------
        pred_segs : np.ndarray of shape (N, 2) or empty (0, 2)
        """
        if pred_gbvn is None:
            return np.empty((0, 2))
        try:
            vid_preds = pred_gbvn.get_group(video_id)
            mask = vid_preds["score"].values >= score_thresh
            if mask.sum() == 0:
                return np.empty((0, 2))
            return vid_preds.loc[mask, ["t-start", "t-end"]].values
        except KeyError:
            return np.empty((0, 2))

    def _compute_count_mae(self):
        """Compute Mean Absolute Error of event counts per video.

        For each video in the subset (including videos with 0 GT events),
        compares the number of ground truth events to the number of predicted
        events (filtered by score threshold). Returns MAE at each configured
        score threshold, plus additional diagnostic metrics.

        Returns
        -------
        results : dict
            Dictionary with keys like 'count_MAE@0.3', 'count_MAE@0.5', etc.
            Also includes 'count_MAE_best' (minimum across thresholds),
            'zero_event_accuracy@threshold' (fraction of 0-event videos
            correctly predicted as 0), and 'total_videos_in_subset'.
        """
        results = {}
        results["total_videos_in_subset"] = len(self.all_video_ids)

        # Pre-group predictions by video-id for efficiency
        if not self.prediction.empty:
            pred_gbvn = self.prediction.groupby("video-id")
        else:
            pred_gbvn = None

        best_mae = float("inf")
        for thresh in self.count_score_thresholds:
            abs_errors = []
            zero_gt_correct = 0
            zero_gt_total = 0

            for video_id in self.all_video_ids:
                gt_count = self.gt_count_per_video.get(video_id, 0)

                # Count predictions above threshold for this video
                pred_count = 0
                if pred_gbvn is not None:
                    try:
                        vid_preds = pred_gbvn.get_group(video_id)
                        pred_count = int((vid_preds["score"] >= thresh).sum())
                    except KeyError:
                        pred_count = 0

                abs_errors.append(abs(gt_count - pred_count))

                # Track accuracy on zero-event videos
                if gt_count == 0:
                    zero_gt_total += 1
                    if pred_count == 0:
                        zero_gt_correct += 1

            mae = np.mean(abs_errors)
            results[f"count_MAE@{thresh}"] = mae

            if zero_gt_total > 0:
                results[f"zero_event_accuracy@{thresh}"] = zero_gt_correct / zero_gt_total
            else:
                results[f"zero_event_accuracy@{thresh}"] = float("nan")

            if mae < best_mae:
                best_mae = mae

        results["count_MAE_best"] = best_mae
        return results

    def logging(self, logger=None):
        if logger == None:
            pprint = print
        else:
            pprint = logger.info

        pprint("Loaded annotations from {} subset.".format(self.subset))
        pprint("Number of ground truth instances: {}".format(len(self.ground_truth)))
        pprint("Number of predictions: {}".format(len(self.prediction)))
        pprint("Fixed threshold for tiou score: {}".format(self.tiou_thresholds))
        pprint("Average-mAP: {:>4.2f} (%)".format(self.average_mAP * 100))
        for tiou, mAP in zip(self.tiou_thresholds, self.mAPs):
            pprint("mAP at tIoU {:.2f} is {:>4.2f}%".format(tiou, mAP * 100))

        # if top_k is not None, print top-kx recall
        if self.top_k is not None:
            pprint("Fixed top-kx results: {}".format(self.top_k))
            for tiou, recall in zip(self.tiou_thresholds, self.mRecall):
                recall_string = ["R{:d} is {:>4.2f}%".format(k, r * 100) for k, r in zip(self.top_k, recall)]
                pprint("Recall at tIoU {:.2f}: {}".format(tiou, ", ".join(recall_string)))

        # Print ethologist-oriented count and recall metrics
        if self.compute_count_metrics and hasattr(self, "count_metrics"):
            pprint("")
            pprint("=" * 70)
            pprint("ETHOLOGIST METRICS: Event Detection Recall & Count Analysis")
            pprint("=" * 70)
            pprint("Videos with events: {}  |  Videos without events: {}".format(
                self.count_metrics["n_positive_videos"],
                self.count_metrics["n_negative_videos"],
            ))
            pprint("")

            for score_thresh in self.count_score_thresholds:
                pprint("-" * 70)
                pprint("Score threshold: {:.2f}".format(score_thresh))
                pprint("-" * 70)

                # Event detection recall
                pprint("  Event Detection Recall (did we catch real events?):")
                for tiou_thresh in self.tiou_thresholds:
                    key = f"event_recall@tIoU{tiou_thresh}@s{score_thresh}"
                    pprint("    tIoU >= {:.2f}: {:>5.1f}%".format(
                        tiou_thresh, self.count_metrics[key] * 100
                    ))

                # Count metrics
                pprint("  Count Metrics (positive-event videos only):")
                pprint("    Count MAE:    {:.2f} events (avg error in predicted count)".format(
                    self.count_metrics[f"count_MAE_positive@s{score_thresh}"]
                ))
                pprint("    Count Recall: {:.1f}% (did we predict enough events?)".format(
                    self.count_metrics[f"count_recall@s{score_thresh}"] * 100
                ))

                # Specificity
                pprint("  Empty Clip Behavior:")
                spec = self.count_metrics[f"zero_event_specificity@s{score_thresh}"]
                far = self.count_metrics[f"false_alarm_rate@s{score_thresh}"]
                if not np.isnan(spec):
                    pprint("    Specificity:      {:.1f}% (empty clips correctly silent)".format(spec * 100))
                    pprint("    False Alarm Rate: {:.2f} predictions/empty clip".format(far))
                pprint("")

            pprint("=" * 70)


def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=np.linspace(0.2, 0.95, 16)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction["score"].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby("video-id")

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():
        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred["video-id"])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[["t-start", "t-end"]].values, this_gt[["t-start", "t-end"]].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]["index"]] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]["index"]] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    ap = np.zeros(len(tiou_thresholds))

    for tidx in range(len(tiou_thresholds)):
        # Computing prec-rec
        this_tp = np.cumsum(tp[tidx, :]).astype(float)
        this_fp = np.cumsum(fp[tidx, :]).astype(float)
        rec = this_tp / npos
        prec = this_tp / (this_tp + this_fp)
        ap[tidx] = interpolated_prec_rec(prec, rec)
    return ap


def compute_topkx_recall_detection(
    ground_truth,
    prediction,
    tiou_thresholds=np.linspace(0.1, 0.5, 5),
    top_k=(1, 5),
):
    """Compute recall (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    top_k: tuple, optional
        Top-kx results of a action category where x stands for the number of
        instances for the action category in the video.
    Outputs
    -------
    recall : float
        Recall score.
    """
    if prediction.empty:
        return np.zeros((len(tiou_thresholds), len(top_k)))

    # Initialize true positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(top_k)))
    n_gts = 0

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby("video-id")
    prediction_gbvn = prediction.groupby("video-id")

    for videoid, _ in ground_truth_gbvn.groups.items():
        ground_truth_videoid = ground_truth_gbvn.get_group(videoid)
        n_gts += len(ground_truth_videoid)
        try:
            prediction_videoid = prediction_gbvn.get_group(videoid)
        except Exception as e:
            continue

        this_gt = ground_truth_videoid.reset_index()
        this_pred = prediction_videoid.reset_index()

        # Sort predictions by decreasing score order.
        score_sort_idx = this_pred["score"].values.argsort()[::-1]
        top_kx_idx = score_sort_idx[: max(top_k) * len(this_gt)]
        tiou_arr = k_segment_iou(
            this_pred[["t-start", "t-end"]].values[top_kx_idx], this_gt[["t-start", "t-end"]].values
        )

        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for kidx, k in enumerate(top_k):
                tiou = tiou_arr[: k * len(this_gt)]
                tp[tidx, kidx] += ((tiou >= tiou_thr).sum(axis=0) > 0).sum()

    recall = tp / n_gts

    return recall


def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.

    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (
        (candidate_segments[:, 1] - candidate_segments[:, 0])
        + (target_segment[1] - target_segment[0])
        - segments_intersection
    )
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union.clip(1e-8)
    return tIoU


def k_segment_iou(target_segments, candidate_segments):
    return np.stack([segment_iou(target_segment, candidate_segments) for target_segment in target_segments])


def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011."""
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap
