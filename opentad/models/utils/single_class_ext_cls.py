# opentad/models/utils/single_class_ext_cls.py

import torch


class SingleClassExtCls:
    """
    Pass-through external classifier for single-class BMN inference.

    BMN is architecturally a proposal generator; its post_processing()
    requires an external classifier to assign semantic labels.  For a
    single-class problem (displacement / no-displacement) we simply
    assign label 0 to every surviving proposal and leave the IoU scores
    untouched.
    """

    def __init__(self, class_name: str = "displacement"):
        self.class_name = class_name

    def __call__(self, video_id, segments, scores):
        """
        Args:
            video_id : str
            segments : Tensor [N, 2]  — start/end in seconds
            scores   : Tensor [N]     — BMN IoU confidence scores
        Returns:
            segments, labels, scores  (labels are all 0)
        """
        labels = torch.zeros(len(scores), dtype=torch.long)
        return segments, labels, scores
