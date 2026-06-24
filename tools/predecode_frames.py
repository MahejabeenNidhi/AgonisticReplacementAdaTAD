#!/usr/bin/env python3
"""
Pre-decode video frames for accelerated AdaTAD training.

Extracts only the frames that LoadFrames (method="resize") would select,
saving them as JPEG files. This eliminates video decoding during training.

Usage:
    python tools/predecode_frames.py \
        --video_dir data/displacement/videos \
        --annotation data/displacement/annotations/TAL_2min_all.json \
        --output_dir data/displacement/frames_192 \
        --short_side 192 \
        --resize_length 160 \
        --scale_factor 2 \
        --num_workers 8 \
        --jpeg_quality 95
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

import cv2

# Use decord for fast video reading
import decord
decord.bridge.set_bridge("native")


def compute_frame_indices(total_frames, resize_length=160, scale_factor=2):
    """
    Compute the exact frame indices that LoadFrames (method='resize') selects.
    This mirrors the logic in opentad/datasets/transforms/end_to_end.py.
    """
    frame_num = resize_length * scale_factor  # e.g., 160 * 2 = 320
    frame_stride = total_frames / frame_num
    frame_idxs = np.arange(
        frame_stride / 2 - 0.5,
        total_frames + frame_stride / 2 - 0.5,
        frame_stride,
    )
    frame_idxs = np.clip(frame_idxs, 0, total_frames - 1).round().astype(int)
    # Ensure we have exactly frame_num indices
    frame_idxs = frame_idxs[:frame_num]
    return frame_idxs


def resize_short_side(img, short_side):
    """Resize image so that the shorter side equals short_side."""
    h, w = img.shape[:2]
    if h <= w:
        new_h = short_side
        new_w = int(round(w * short_side / h))
    else:
        new_w = short_side
        new_h = int(round(h * short_side / w))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def extract_video(video_name, video_dir, output_dir, resize_length,
                  scale_factor, short_side, jpeg_quality):
    """Extract pre-determined frames from a single video."""
    # Find video file
    video_path = os.path.join(video_dir, video_name + ".mp4")
    if not os.path.exists(video_path):
        return f"SKIP (not found): {video_path}"

    out_path = os.path.join(output_dir, video_name)

    # Check if already extracted (resume capability)
    expected_num_frames = resize_length * scale_factor
    if os.path.exists(out_path):
        existing = [f for f in os.listdir(out_path) if f.endswith(".jpg")]
        if len(existing) == expected_num_frames:
            return f"SKIP (already done): {video_name}"

    os.makedirs(out_path, exist_ok=True)

    try:
        # Open video with decord
        vr = decord.VideoReader(video_path, num_threads=1)
        total_frames = len(vr)

        # Compute the exact frame indices
        frame_idxs = compute_frame_indices(total_frames, resize_length, scale_factor)

        # Handle edge case: video shorter than expected
        if len(frame_idxs) < expected_num_frames:
            # Pad by repeating the last index
            pad_count = expected_num_frames - len(frame_idxs)
            frame_idxs = np.concatenate([
                frame_idxs,
                np.full(pad_count, frame_idxs[-1], dtype=int)
            ])

        # Read frames in batch (much faster than one-by-one)
        frames = vr.get_batch(frame_idxs.tolist()).asnumpy()  # (N, H, W, 3) RGB

        # Save each frame
        for i, frame in enumerate(frames):
            # Resize if requested
            if short_side > 0:
                frame = resize_short_side(frame, short_side)

            # Convert RGB -> BGR for cv2.imwrite
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            filepath = os.path.join(out_path, f"frame_{i:06d}.jpg")
            cv2.imwrite(filepath, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

        # Save metadata for verification
        meta = {
            "video_name": video_name,
            "total_frames_in_video": int(total_frames),
            "extracted_frame_count": int(expected_num_frames),
            "resize_length": resize_length,
            "scale_factor": scale_factor,
            "short_side": short_side,
            "frame_indices": frame_idxs.tolist(),
        }
        meta_path = os.path.join(out_path, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return f"OK: {video_name} ({total_frames} total -> {expected_num_frames} extracted)"

    except Exception as e:
        return f"ERROR: {video_name} - {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Pre-decode video frames for AdaTAD")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing video files")
    parser.add_argument("--annotation", type=str, required=True,
                        help="Path to annotation JSON")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for pre-decoded frames")
    parser.add_argument("--short_side", type=int, default=192,
                        help="Resize shortest side to this value. Set 0 for original resolution.")
    parser.add_argument("--resize_length", type=int, default=160,
                        help="Must match config's resize_length")
    parser.add_argument("--scale_factor", type=int, default=2,
                        help="Must match config's LoadFrames scale_factor")
    parser.add_argument("--jpeg_quality", type=int, default=95,
                        help="JPEG quality (1-100)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel workers")
    parser.add_argument("--subset", type=str, default=None,
                        help="Only extract a specific subset (training/validation/testing)")
    args = parser.parse_args()

    # Load annotation to get list of all videos
    with open(args.annotation, "r") as f:
        ann_data = json.load(f)["database"]

    # Collect video names
    video_names = []
    for clip_name, clip_info in ann_data.items():
        if args.subset is not None and clip_info.get("subset") != args.subset:
            continue
        video_names.append(clip_name)

    print(f"=" * 70)
    print(f"Pre-decoding video frames")
    print(f"=" * 70)
    print(f"  Video directory:   {args.video_dir}")
    print(f"  Output directory:  {args.output_dir}")
    print(f"  Total videos:      {len(video_names)}")
    print(f"  Resize length:     {args.resize_length}")
    print(f"  Scale factor:      {args.scale_factor}")
    print(f"  Frames per video:  {args.resize_length * args.scale_factor}")
    print(f"  Short side resize: {args.short_side if args.short_side > 0 else 'Original'}")
    print(f"  JPEG quality:      {args.jpeg_quality}")
    print(f"  Workers:           {args.num_workers}")
    print(f"=" * 70)

    # Estimate storage
    frames_per_video = args.resize_length * args.scale_factor
    if args.short_side > 0 and args.short_side <= 200:
        est_per_frame_kb = 12  # ~12KB for 192px short side
    elif args.short_side > 200 and args.short_side <= 300:
        est_per_frame_kb = 25
    else:
        est_per_frame_kb = 80  # 720p
    est_total_gb = len(video_names) * frames_per_video * est_per_frame_kb / 1024 / 1024
    print(f"  Estimated storage: ~{est_total_gb:.1f} GB")
    print(f"=" * 70)
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    # Create partial function with fixed args
    extract_fn = partial(
        extract_video,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        resize_length=args.resize_length,
        scale_factor=args.scale_factor,
        short_side=args.short_side,
        jpeg_quality=args.jpeg_quality,
    )

    # Process in parallel
    if args.num_workers > 1:
        with Pool(processes=args.num_workers) as pool:
            results = list(pool.imap_unordered(extract_fn, video_names))
    else:
        results = [extract_fn(vn) for vn in video_names]

    # Summary
    ok_count = sum(1 for r in results if r.startswith("OK"))
    skip_count = sum(1 for r in results if r.startswith("SKIP"))
    error_count = sum(1 for r in results if r.startswith("ERROR"))

    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"  Extracted: {ok_count}")
    print(f"  Skipped:   {skip_count}")
    print(f"  Errors:    {error_count}")
    print(f"{'=' * 70}")

    # Print errors if any
    errors = [r for r in results if r.startswith("ERROR")]
    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"  {e}")

    # Compute actual disk usage
    actual_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, _, filenames in os.walk(args.output_dir)
        for filename in filenames
    )
    print(f"\nActual disk usage: {actual_size / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
