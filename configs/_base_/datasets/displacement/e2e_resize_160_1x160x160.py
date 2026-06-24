# Base dataset config for 2-min displacement clips
# Temporal resolution: 120s / 152 tokens = 0.789 s/token

dataset_type = "AnetResizeDataset"
annotation_path = "data/displacement/annotations/annotations.json"
class_map = "data/displacement/annotations/class_map.txt"
data_path = "data/displacement/videos"
block_list = None

resize_length = 160

dataset = dict(
    train=dict(
        type="AnetResizeDataset",
        ann_file=annotation_path,
        subset_name="training",
        data_path=data_path,
        class_map=class_map,
        filter_gt=True,
        class_agnostic=False,
        test_mode=False,
        resize_length=160,
        sample_stride=1,
        empty_clip_ratio=0.0,
        pipeline=None,
    ),
    val=dict(
        type="AnetResizeDataset",
        ann_file=annotation_path,
        subset_name="validation",
        data_path=data_path,
        class_map=class_map,
        filter_gt=True,
        class_agnostic=False,
        test_mode=False,
        resize_length=160,
        sample_stride=1,
        empty_clip_ratio=0.0,
        pipeline=None,
    ),
    test=dict(
        type="AnetResizeDataset",
        ann_file=annotation_path,
        subset_name="testing",
        data_path=data_path,
        class_map=class_map,
        filter_gt=False,
        class_agnostic=False,
        test_mode=True,
        resize_length=160,
        sample_stride=1,
        pipeline=None,
    ),
)

evaluation = dict(
    type="mAP",
    subset="testing",
    tiou_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    ground_truth_filename=annotation_path,
    blocked_videos=None,
    compute_count_metrics=True,
    count_score_thresholds=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    compute_period_counts=True,
    period_count_score_thresholds=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    period_nms_iou_threshold=0.3,
)
