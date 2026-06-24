import json
import os
import numpy as np
from ..builder import DATASETS, get_class_index
from mmengine.dataset import Compose


@DATASETS.register_module()
class ResizeDataset:
    def __init__(
        self,
        ann_file,  # path of the annotation json file
        subset_name,  # name of the subset, such as training, validation, testing
        data_path,  # folder path of the raw video / pre-extracted feature
        pipeline,  # data pipeline
        class_map,  # path of the class map, convert the class id to category name
        filter_gt=False,  # if True, filter out those gt has the scale smaller than 0.01
        class_agnostic=False,  # if True, the class index will be replaced by 0
        block_list=None,  # some videos might be missed in the features or videos, we need to block them
        test_mode=False,  # if True, running on test mode with no annotation
        resize_length=128,  # the length of the resized video
        sample_stride=1,  # if you want to extract the feature[::sample_stride]
        include_empty_clips=False,  # legacy flag; superseded by empty_clip_ratio
        empty_clip_ratio=0.0,  # NEW: ratio of empty clips to non-empty clips per epoch
        logger=None,
    ):
        super(ResizeDataset, self).__init__()
        # basic settings
        self.data_path = data_path
        self.block_list = block_list
        self.ann_file = ann_file
        self.subset_name = subset_name
        self.logger = logger.info if logger is not None else print
        self.class_map = self.get_class_map(class_map)
        self.class_agnostic = class_agnostic
        self.filter_gt = filter_gt
        self.test_mode = test_mode
        self.resize_length = resize_length
        self.sample_stride = sample_stride
        self.include_empty_clips = include_empty_clips
        self.empty_clip_ratio = empty_clip_ratio
        self.pipeline = Compose(pipeline)

        # Internal storage: positive clips and empty clips are kept separately
        self._positive_data_list = []
        self._empty_data_list = []
        self._current_epoch = 0
        self._base_seed = 42  # deterministic across ranks

        self.get_dataset()

        # Compose the initial data_list (epoch 0)
        self._compose_data_list()
        self.logger(
            f"{self.subset_name} subset: {len(self._positive_data_list)} positive clips, "
            f"{len(self._empty_data_list)} empty clips, "
            f"{len(self.data_list)} used this epoch (empty_clip_ratio={self.empty_clip_ratio})"
        )

    def get_dataset(self):
        with open(self.ann_file, "r") as f:
            anno_database = json.load(f)["database"]

        # some videos might be missed in the features or videos, we need to block them
        if self.block_list is not None:
            if isinstance(self.block_list, list):
                blocked_videos = self.block_list
            else:
                with open(self.block_list, "r") as f:
                    blocked_videos = [line.rstrip("\n") for line in f]
        else:
            blocked_videos = []

        self._positive_data_list = []
        self._empty_data_list = []

        for video_name, video_info in anno_database.items():
            if (video_name in blocked_videos) or (
                video_info["subset"] not in self.subset_name
            ):
                continue

            if self.test_mode:
                # In test mode, all clips are kept regardless (no GT needed)
                video_anno = {}
                self._positive_data_list.append([video_name, video_info, video_anno])
                continue

            video_anno = self.get_gt(video_info)

            if video_anno is None:
                # This is an empty (background-only) clip
                empty_anno = dict(
                    gt_segments=np.zeros((0, 2), dtype=np.float32),
                    gt_labels=np.zeros((0,), dtype=np.int32),
                )
                self._empty_data_list.append([video_name, video_info, empty_anno])
            else:
                self._positive_data_list.append([video_name, video_info, video_anno])

        assert len(self._positive_data_list) > 0 or len(self._empty_data_list) > 0, (
            f"No data found in {self.subset_name} subset."
        )

    def _compose_data_list(self):
        """Compose self.data_list from positive clips + sampled empty clips."""
        # Always include all positive clips
        composed = list(self._positive_data_list)

        num_positive = len(self._positive_data_list)
        num_empty_available = len(self._empty_data_list)

        if num_empty_available == 0 or self.test_mode:
            # Nothing to add or test mode (all clips already in _positive_data_list)
            self.data_list = composed
            return

        if self.empty_clip_ratio < 0:
            # Sentinel value (-1 or any negative): include ALL empty clips
            composed.extend(self._empty_data_list)
        elif self.empty_clip_ratio == 0.0:
            # No empty clips
            pass
        else:
            # Sample empty clips: num_to_sample = ratio * num_positive
            num_to_sample = int(round(self.empty_clip_ratio * num_positive))
            num_to_sample = max(0, min(num_to_sample, num_empty_available))

            if num_to_sample > 0:
                # Use a deterministic RNG seeded by epoch so all ranks agree
                rng = np.random.RandomState(self._base_seed + self._current_epoch)
                sampled_indices = rng.choice(
                    num_empty_available, size=num_to_sample, replace=False
                )
                for idx in sampled_indices:
                    composed.append(self._empty_data_list[idx])

        self.data_list = composed

    def set_epoch(self, epoch):
        """Called before each epoch to re-sample empty clips.

        Must be called with the same epoch value on all DDP ranks to keep
        the dataset length consistent across processes.
        """
        if self._current_epoch != epoch or epoch == 0:
            self._current_epoch = epoch
            self._compose_data_list()

    def get_class_map(self, class_map_path):
        if not os.path.exists(class_map_path):
            class_map = get_class_index(self.ann_file, class_map_path)
            self.logger(f"Class map is saved in {class_map_path}, total {len(class_map)} classes.")
        else:
            with open(class_map_path, "r", encoding="utf8") as f:
                lines = f.readlines()
            class_map = [item.rstrip("\n") for item in lines]
        return class_map

    def get_gt(self):
        pass

    def __getitem__(self):
        pass

    def __len__(self):
        return len(self.data_list)