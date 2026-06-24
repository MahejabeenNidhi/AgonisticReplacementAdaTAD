"""
Pipeline transforms for loading pre-decoded video frames.
Replaces PrepareVideoInfo + DecordInit + DecordDecode with fast JPEG loading.
"""

import os
import numpy as np
import cv2
from ..builder import PIPELINES

@PIPELINES.register_module()
class CodecSimulationJitter:
    """
    Simulates the pixel-level non-determinism of video codec decoding.

    When reading pre-decoded JPEG frames, every epoch sees identical pixels.
    Video codecs (H.264/H.265) produce subtly different outputs across reads
    due to threading, B-frame seeking, and rounding in motion compensation.
    This transform injects equivalent controlled noise to prevent the model
    from memorizing fixed pixel patterns of empty clips.

    Place immediately after DecodePredecodedFrames, before spatial
    augmentations.

    Args:
        noise_std_range (tuple): Range [lo, hi] for per-clip Gaussian noise σ.
            One σ is sampled per clip; all frames share the same σ but get
            independent noise realizations. Default: (0.5, 2.5).
        channel_shift_range (float): Maximum per-channel uniform shift.
            Simulates white-balance/chroma rounding differences. Default: 1.5.
        jpeg_resample_prob (float): Per-frame probability of random JPEG
            re-compression. Simulates DCT blocking artifact variation.
            Default: 0.0 (disabled for speed; set to 0.15-0.3 if GPU-bound).
        jpeg_quality_range (tuple): Quality range for re-compression.
            Default: (85, 96).
        temporal_smooth_prob (float): Probability of blending adjacent frames
            slightly, simulating temporal interpolation artifacts from B-frame
            decoding. Default: 0.1.
        temporal_smooth_alpha (tuple): Blend alpha range [lo, hi].
            Default: (0.02, 0.08).
    """

    def __init__(
            self,
            noise_std_range=(0.5, 2.5),
            channel_shift_range=1.5,
            jpeg_resample_prob=0.0,
            jpeg_quality_range=(85, 96),
            temporal_smooth_prob=0.1,
            temporal_smooth_alpha=(0.02, 0.08),
    ):
        self.noise_std_range = noise_std_range
        self.channel_shift_range = channel_shift_range
        self.jpeg_resample_prob = jpeg_resample_prob
        self.jpeg_quality_range = jpeg_quality_range
        self.temporal_smooth_prob = temporal_smooth_prob
        self.temporal_smooth_alpha = temporal_smooth_alpha

    def __call__(self, results):
        imgs = results["imgs"]  # list of (H, W, 3) uint8 RGB arrays
        num_frames = len(imgs)

        # Sample per-clip noise level (all frames share the same σ)
        noise_std = np.random.uniform(*self.noise_std_range)

        # Sample per-clip channel shift (constant across frames, as codec
        # rounding errors are consistent within a decode call)
        channel_shift = np.random.uniform(
            -self.channel_shift_range, self.channel_shift_range, size=(1, 1, 3)
        ).astype(np.float32)

        # Decide temporal smoothing indices upfront
        do_temporal = (
                self.temporal_smooth_prob > 0
                and np.random.rand() < self.temporal_smooth_prob
                and num_frames > 1
        )
        if do_temporal:
            alpha = np.random.uniform(*self.temporal_smooth_alpha)

        new_imgs = []
        for i, img in enumerate(imgs):
            img_f = img.astype(np.float32)

            # 1. Gaussian noise (independent per frame, shared σ)
            if noise_std > 0:
                noise = np.random.randn(*img_f.shape).astype(np.float32) * noise_std
                img_f += noise

            # 2. Channel shift (shared per clip)
            img_f += channel_shift

            # 3. Temporal smoothing with adjacent frame
            if do_temporal and i > 0:
                prev_f = imgs[i - 1].astype(np.float32)
                img_f = (1.0 - alpha) * img_f + alpha * prev_f

            # 4. Optional JPEG re-compression (expensive, use sparingly)
            if self.jpeg_resample_prob > 0 and np.random.rand() < self.jpeg_resample_prob:
                quality = np.random.randint(*self.jpeg_quality_range)
                img_uint8 = np.clip(img_f, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
                _, enc = cv2.imencode(
                    ".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]
                )
                img_f = cv2.cvtColor(
                    cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
                ).astype(np.float32)

            new_imgs.append(np.clip(img_f, 0, 255).astype(np.uint8))

        results["imgs"] = new_imgs
        return results

@PIPELINES.register_module()
class PreparePredecodedFrames:
    """
    Replaces PrepareVideoInfo + DecordInit for pre-decoded frame directories.

    Sets all metadata keys that DecordInit + PrepareVideoInfo normally set,
    ensuring downstream transforms have consistent state.

    Args:
        frames_dir (str): Root directory containing per-video frame folders.
        num_frames (int): Number of pre-extracted frames per video
            (= resize_length * scale_factor).
        fps (float): Original video FPS (used for metadata).
    """

    def __init__(self, frames_dir, num_frames=320, fps=20.0):
        self.frames_dir = frames_dir
        self.num_frames = num_frames
        self.fps = fps

    def __call__(self, results):
        video_name = results["video_name"]
        frame_dir = os.path.join(self.frames_dir, video_name)

        # Verify the frame directory exists (fail fast for missing clips)
        if not os.path.isdir(frame_dir):
            raise FileNotFoundError(
                f"Pre-decoded frame directory not found: {frame_dir}. "
                f"Run tools/predecode_frames.py first."
            )

        results["frame_dir"] = frame_dir

        # Set total_frames = num_frames so LoadFrames generates indices [0..N-1]
        results["total_frames"] = self.num_frames
        results["avg_fps"] = self.fps

        # Set modality explicitly (PrepareVideoInfo normally does this)
        results["modality"] = "RGB"

        # Set start_index for mmaction2 compatibility
        results["start_index"] = 0

        return results


@PIPELINES.register_module()
class DecodePredecodedFrames:
    """
    Reads pre-decoded JPEG frames from disk. Replaces DecordDecode.

    Expects results["frame_dir"] and results["frame_inds"] to be set
    (by PreparePredecodedFrames and LoadFrames respectively).

    Outputs results["imgs"] as a list of numpy arrays, each (H, W, 3)
    uint8 RGB, matching DecordDecode's output format.

    Args:
        filename_tmpl (str): Template for frame filenames (0-indexed).
        temporal_jitter (int): Maximum frame index offset applied randomly
            per frame. Simulates the slight temporal misalignment that occurs
            when video codecs decode from different seek points. The GT
            misalignment is negligible for jitter <= 2 (shifts < 0.5s for
            typical stride). Default: 0 (disabled).
    """

    def __init__(self, filename_tmpl="frame_{:06d}.jpg", temporal_jitter=0):
        self.filename_tmpl = filename_tmpl
        self.temporal_jitter = temporal_jitter

    def __call__(self, results):
        frame_dir = results["frame_dir"]
        frame_inds = results["frame_inds"]  # shape (N,), set by LoadFrames

        # Determine valid index range from available frames
        max_idx = results.get("total_frames", len(frame_inds)) - 1

        # Apply temporal jitter during training only
        if self.temporal_jitter > 0 and results.get("gt_segments") is not None:
            jitter = np.random.randint(
                -self.temporal_jitter, self.temporal_jitter + 1,
                size=frame_inds.shape,
            )
            frame_inds = np.clip(frame_inds + jitter, 0, max_idx)

        imgs = []
        for idx in frame_inds:
            filepath = os.path.join(frame_dir, self.filename_tmpl.format(int(idx)))
            img_bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise FileNotFoundError(
                    f"Could not read frame: {filepath}. "
                    f"Check that pre-decoding completed successfully."
                )
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            imgs.append(img_rgb)

        results["imgs"] = imgs  # list format, matching DecordDecode
        results["original_shape"] = imgs[0].shape[:2]
        results["img_shape"] = imgs[0].shape[:2]
        return results
