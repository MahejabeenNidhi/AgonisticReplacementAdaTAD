"""
python check_adatad_compat.py /media/D2/public/mae/mmaction2/work_dirs/videomaev2_rgb_BNfrozen/best_acc_top1_epoch_20.pth
"""

import argparse
import sys
import torch
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# Known VideoMAEv2 / VideoMAE architecture configurations
# ─────────────────────────────────────────────────────────────────────────────
ARCH_CONFIGS = {
    "small": {
        "embed_dims": 384,
        "depth": 12,
        "num_heads": 6,
        "patch_size": 16,
        "mlp_ratio": 4.0,
        "approx_params": "22M",
    },
    "base": {
        "embed_dims": 768,
        "depth": 12,
        "num_heads": 12,
        "patch_size": 16,
        "mlp_ratio": 4.0,
        "approx_params": "86M",
    },
    "large": {
        "embed_dims": 1024,
        "depth": 24,
        "num_heads": 16,
        "patch_size": 16,
        "mlp_ratio": 4.0,
        "approx_params": "307M",
    },
    "huge": {
        "embed_dims": 1280,
        "depth": 32,
        "num_heads": 16,
        "patch_size": 16,
        "mlp_ratio": 4.0,
        "approx_params": "632M",
    },
    "giant": {
        "embed_dims": 1408,
        "depth": 40,
        "num_heads": 16,
        "patch_size": 14,
        "mlp_ratio": round(48 / 11, 6),
        "approx_params": "~1B",
    },
}

# Keys that AdaTAD's VisionTransformerAdapter expects from the backbone
# (after the 'backbone.' prefix has been stripped)
EXPECTED_ADATAD_KEYS = [
    "patch_embed.projection.weight",
    "patch_embed.projection.bias",
    "blocks.0.norm1.weight",
    "blocks.0.norm1.bias",
    "blocks.0.attn.qkv.weight",
    "blocks.0.attn.proj.weight",
    "blocks.0.norm2.weight",
    "blocks.0.mlp.layers.0.0.weight",   # fc1 in mmaction2 naming
    "blocks.0.mlp.layers.1.weight",     # fc2 in mmaction2 naming
]

# Keys that signal the checkpoint is from the ORIGINAL VideoMAEv2 repo
# and needs conversion before AdaTAD can use it
ORIGINAL_REPO_SIGNALS = [
    ("fc1",                  "FFN first layer — needs renaming to layers.0.0"),
    ("fc2",                  "FFN second layer — needs renaming to layers.1"),
    ("patch_embed.proj.",    "Patch embed — needs renaming to patch_embed.projection"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def section(title: str):
    bar = "─" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def ok(msg):   print(f"  ✓  {msg}")
def warn(msg): print(f"  ⚠  {msg}")
def fail(msg): print(f"  ✗  {msg}")


def load_checkpoint(path: str) -> dict:
    """Load a .pth file and return its contents as a dict."""
    try:
        ckpt = torch.load(path, map_location="cpu")
        return ckpt
    except Exception as exc:
        fail(f"Could not load file: {exc}")
        sys.exit(1)


def extract_state_dict(ckpt: dict) -> tuple[dict, dict]:
    """
    Return (state_dict, meta) where state_dict is the flat weight dict
    and meta holds any surrounding metadata.
    """
    meta = {}
    if not isinstance(ckpt, dict):
        fail("Checkpoint is not a Python dict — unexpected format.")
        sys.exit(1)

    top_keys = set(ckpt.keys())

    # mmaction2 / mmengine saved checkpoint
    if "state_dict" in top_keys:
        meta["epoch"]    = ckpt.get("epoch", "unknown")
        meta["mmaction"] = True
        if "meta" in ckpt and isinstance(ckpt["meta"], dict):
            meta.update({f"meta.{k}": v for k, v in ckpt["meta"].items()})
        return ckpt["state_dict"], meta

    # Original VideoMAEv2 repo (DDP-wrapped)
    if "module" in top_keys:
        meta["original_repo"] = True
        return ckpt["module"], meta

    # Some repos use "model"
    if "model" in top_keys:
        meta["model_key"] = True
        return ckpt["model"], meta

    # Flat state dict
    meta["flat"] = True
    return ckpt, meta


def detect_format(sd: dict) -> str:
    """
    Returns one of:
        'mmaction2'          — already compatible with AdaTAD
        'original_repo'      — needs conversion
        'unknown'
    """
    keys = list(sd.keys())

    has_backbone_prefix  = any(k.startswith("backbone.") for k in keys)
    has_fc1_naming       = any("fc1" in k for k in keys)
    has_old_patch_embed  = any("patch_embed.proj." in k for k in keys)
    has_module_prefix    = any(k.startswith("module.") for k in keys)

    if has_fc1_naming or has_old_patch_embed or has_module_prefix:
        return "original_repo"

    # mmaction2 exports always have backbone. prefix (full Recognizer3D)
    # or may be just the backbone weights directly
    if has_backbone_prefix:
        return "mmaction2"

    # Could be a bare backbone state dict saved from mmaction2
    # Check for mmaction2-style FFN naming
    has_layers_naming = any("mlp.layers" in k for k in keys)
    has_projection     = any("patch_embed.projection" in k for k in keys)
    if has_layers_naming or has_projection:
        return "mmaction2"

    return "unknown"


def strip_to_backbone(sd: dict, fmt: str) -> dict:
    """Return a state dict with only backbone weights, prefix stripped."""
    if fmt == "mmaction2":
        backbone = {k[len("backbone."):]: v
                    for k, v in sd.items()
                    if k.startswith("backbone.")}
        if backbone:
            return backbone
        # Already bare backbone weights
        return sd
    # For unknown/original: return as-is for analysis
    return sd


def detect_arch(backbone_sd: dict) -> dict:
    """
    Infer architecture (embed_dims, depth, patch_size, num_heads) directly
    from tensor shapes in the state dict.
    """
    embed_dims = None
    patch_size = None
    depth      = 0
    num_heads  = None

    # embed_dims and patch_size from patch_embed projection kernel
    for key in ("patch_embed.projection.weight", "patch_embed.proj.weight"):
        if key in backbone_sd:
            w = backbone_sd[key]       # shape: [embed_dims, in_ch, T, pH, pW]
            embed_dims = w.shape[0]
            patch_size = w.shape[-1]   # spatial patch size
            break

    # depth from number of distinct block indices
    block_ids = set()
    for k in backbone_sd:
        if k.startswith("blocks."):
            parts = k.split(".")
            try:
                block_ids.add(int(parts[1]))
            except (IndexError, ValueError):
                pass
    depth = len(block_ids)

    # num_heads: infer from q_bias shape  ─  q_bias is [embed_dims]
    # Can't get it directly, so derive from known configs
    for arch_name, cfg in ARCH_CONFIGS.items():
        if embed_dims == cfg["embed_dims"] and depth == cfg["depth"]:
            num_heads = cfg["num_heads"]
            break

    # Match to known architecture
    detected = None
    for arch_name, cfg in ARCH_CONFIGS.items():
        dims_match       = embed_dims == cfg["embed_dims"]
        depth_match      = depth      == cfg["depth"]
        patch_match      = (patch_size is None) or (patch_size == cfg["patch_size"])
        if dims_match and depth_match and patch_match:
            detected = arch_name
            break

    return {
        "embed_dims":    embed_dims,
        "depth":         depth,
        "patch_size":    patch_size,
        "num_heads":     num_heads,
        "detected_arch": detected,
    }


def count_params(sd: dict) -> int:
    return sum(v.numel() for v in sd.values() if isinstance(v, torch.Tensor))


def check_original_repo_signals(sd: dict) -> list[tuple[str, str]]:
    hits = []
    for signal, explanation in ORIGINAL_REPO_SIGNALS:
        if any(signal in k for k in sd.keys()):
            hits.append((signal, explanation))
    return hits


def check_expected_keys(backbone_sd: dict) -> tuple[list, list]:
    found   = [k for k in EXPECTED_ADATAD_KEYS if k in backbone_sd]
    missing = [k for k in EXPECTED_ADATAD_KEYS if k not in backbone_sd]
    return found, missing


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Check a VideoMAEv2 .pth file for AdaTAD compatibility."
    )
    parser.add_argument("checkpoint", help="Path to the .pth file")
    parser.add_argument(
        "--show-keys", type=int, default=20, metavar="N",
        help="How many keys to print in the key preview (default: 20)"
    )
    parser.add_argument(
        "--all-keys", action="store_true",
        help="Print every key in the state dict"
    )
    args = parser.parse_args()

    # ── 1. Load ───────────────────────────────────────────────────────────────
    section("Loading Checkpoint")
    print(f"  File: {args.checkpoint}\n")
    raw = load_checkpoint(args.checkpoint)
    ok("File loaded successfully")

    state_dict, meta = extract_state_dict(raw)
    ok(f"State dict extracted  ({len(state_dict):,} keys)")
    if meta:
        for k, v in meta.items():
            print(f"       {k}: {v}")

    # ── 2. Key preview ────────────────────────────────────────────────────────
    section("Key Preview")
    n_show = len(state_dict) if args.all_keys else min(args.show_keys, len(state_dict))
    print(f"  Showing {n_show} / {len(state_dict)} keys:\n")
    fmt_row = "  {:<4}  {:<72}  {}"
    print(fmt_row.format("idx", "key", "shape"))
    print("  " + "─"*90)
    for i, (k, v) in enumerate(list(state_dict.items())[:n_show]):
        shape = str(list(v.shape)) if isinstance(v, torch.Tensor) else type(v).__name__
        print(fmt_row.format(i, k, shape))
    if not args.all_keys and len(state_dict) > n_show:
        print(f"\n  ... ({len(state_dict) - n_show} more keys hidden, use --all-keys to show all)")

    # ── 3. Format detection ───────────────────────────────────────────────────
    section("Checkpoint Format")
    fmt = detect_format(state_dict)

    FORMAT_LABELS = {
        "mmaction2":     "mmaction2 / mmengine  →  compatible with AdaTAD",
        "original_repo": "Original VideoMAEv2 repository  →  conversion required",
        "unknown":       "Unknown format  →  manual inspection needed",
    }
    print(f"  Detected: {FORMAT_LABELS.get(fmt, fmt)}\n")

    has_backbone_prefix = any(k.startswith("backbone.") for k in state_dict)
    has_cls_head        = any(k.startswith("cls_head.")  for k in state_dict)
    has_data_pre        = any(k.startswith("data_preprocessor.") for k in state_dict)

    ok(f"backbone. prefix present : {has_backbone_prefix}")
    ok(f"cls_head. present        : {has_cls_head}  (expected for full Recognizer3D)")
    ok(f"data_preprocessor present: {has_data_pre}")

    # Flag original-repo naming issues
    signals = check_original_repo_signals(state_dict)
    if signals:
        print()
        for sig, explanation in signals:
            fail(f"Found '{sig}' keys  →  {explanation}")

    # ── 4. Architecture ───────────────────────────────────────────────────────
    section("Architecture Detection")
    backbone_sd = strip_to_backbone(state_dict, fmt)
    arch        = detect_arch(backbone_sd)

    print(f"  {'embed_dims':<18}: {arch['embed_dims']}")
    print(f"  {'depth':<18}: {arch['depth']}")
    print(f"  {'patch_size':<18}: {arch['patch_size']}")
    print(f"  {'num_heads':<18}: {arch['num_heads']}")

    if arch["detected_arch"]:
        name = arch["detected_arch"]
        cfg  = ARCH_CONFIGS[name]
        print(f"\n  {'─'*40}")
        ok(f"Model size identified: VideoMAEv2-{name.upper()}")
        print(f"  {'─'*40}")
        print(f"\n  {'Approximate params':<28}: {cfg['approx_params']}")
        print(f"  {'embed_dims':<28}: {cfg['embed_dims']}")
        print(f"  {'depth':<28}: {cfg['depth']}")
        print(f"  {'num_heads':<28}: {cfg['num_heads']}")
        print(f"  {'patch_size':<28}: {cfg['patch_size']}")
        print(f"  {'mlp_ratio':<28}: {cfg['mlp_ratio']}")
    else:
        warn("Could not match to a known architecture.")
        warn("Check embed_dims and depth manually against ARCH_CONFIGS.")

    # ── 5. Parameter count ────────────────────────────────────────────────────
    section("Parameter Count")
    total_params   = count_params(state_dict)
    backbone_params = count_params(backbone_sd)

    print(f"  {'Total in file':<32}: {total_params:>14,}  ({total_params/1e6:.1f} M)")
    if backbone_sd is not state_dict:
        non_bb = total_params - backbone_params
        print(f"  {'Backbone only':<32}: {backbone_params:>14,}  ({backbone_params/1e6:.1f} M)")
        print(f"  {'Non-backbone (head, etc.)':<32}: {non_bb:>14,}  ({non_bb/1e6:.1f} M)")

    # ── 6. AdaTAD key compatibility ───────────────────────────────────────────
    section("AdaTAD Key Compatibility")
    found, missing = check_expected_keys(backbone_sd)

    has_q_bias = any("q_bias" in k for k in backbone_sd)
    has_v_bias = any("v_bias" in k for k in backbone_sd)
    has_fc_norm = "fc_norm.weight" in backbone_sd
    has_norm    = "norm.weight" in backbone_sd

    print("  Expected backbone keys:\n")
    for k in found:
        ok(k)
    for k in missing:
        fail(k)

    print()
    ok(f"q_bias present  : {has_q_bias}  (required for qkv_bias=True)")
    ok(f"v_bias present  : {has_v_bias}  (required for qkv_bias=True)")
    print()
    ok(f"fc_norm present : {has_fc_norm}  (use_mean_pooling=True path)")
    ok(f"norm present    : {has_norm}     (use_mean_pooling=False path)")

    # ── 7. Final verdict ──────────────────────────────────────────────────────
    section("Final Verdict")

    critical_issues = bool(signals) or (fmt == "original_repo")
    key_issues      = len(missing) > 3   # more than minor differences

    if fmt == "mmaction2" and not critical_issues and not key_issues:
        print("""
  ╔══════════════════════════════════════════════════════╗
  ║  ✓  READY — checkpoint is compatible with AdaTAD    ║
  ╚══════════════════════════════════════════════════════╝
""")
    elif fmt == "mmaction2" and not critical_issues and key_issues:
        print("""
  ╔══════════════════════════════════════════════════════╗
  ║  ⚠  LIKELY OK — format is correct but some expected ║
  ║     keys are absent. Verify your config matches the  ║
  ║     actual architecture before training.             ║
  ╚══════════════════════════════════════════════════════╝
""")
    elif fmt == "original_repo":
        print("""
  ╔══════════════════════════════════════════════════════╗
  ║  ✗  NEEDS CONVERSION before use with AdaTAD         ║
  ╚══════════════════════════════════════════════════════╝
""")
        if arch["detected_arch"] == "giant":
            print("  Run:  python tools/model_converters/convert_videomaev2.py \\")
            print("              <in.pth> <out.pth>\n")
        else:
            arch_flag = arch["detected_arch"] or "<arch>"
            print("  Run:  python tools/model_converters/convert_videomae_finetuned.py \\")
            print(f"              <in.pth> <out.pth> --arch {arch_flag}\n")
    else:
        print("""
  ╔══════════════════════════════════════════════════════╗
  ║  ⚠  UNKNOWN — manual inspection required            ║
  ╚══════════════════════════════════════════════════════╝
""")

    # ── 8. Suggested AdaTAD config snippet ───────────────────────────────────
    if arch["detected_arch"]:
        name = arch["detected_arch"]
        cfg  = ARCH_CONFIGS[name]
        fname = args.checkpoint.split("/")[-1]
        section(f"Suggested AdaTAD Config Snippet  (VideoMAEv2-{name.upper()})")
        print(f"""
  model = dict(
      backbone=dict(
          backbone=dict(
              patch_size={cfg['patch_size']},
              embed_dims={cfg['embed_dims']},
              depth={cfg['depth']},
              num_heads={cfg['num_heads']},
              mlp_ratio={cfg['mlp_ratio']},
              adapter_index=list(range({cfg['depth']})),
          ),
          custom=dict(pretrain="pretrained/{fname}"),
      ),
      projection=dict(in_channels={cfg['embed_dims']}),
  )
""")


if __name__ == "__main__":
    main()
