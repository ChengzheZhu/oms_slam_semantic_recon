#!/usr/bin/env python3
"""
04_sam3_mask.py — Pre-compute SAM3 raw mask cache for all extracted frames (L1 cache).

Single-pass approach (image encoded once per frame, two prompts share the encoding):
  Prompt 1: "individual stone" → stone masks
  Prompt 2: "QR code"          → QR masks

QR hole recovery (CPU, post-process):
  QR codes attached to stones appear as holes in the stone mask.
  For each QR mask, the immediate ring (dilation - QR) is checked against all
  stone masks.  If one stone mask covers > --qr_min_coverage of the ring, the
  QR is enclosed by that stone and its pixels are OR-merged into the stone mask.

Writes:
  <frames_dir>/sam3_mask_cache_conf_<c>/          — raw stone masks (L1)
  <frames_dir>/sam3_qr_cache_conf_<c>/            — raw QR masks
  <frames_dir>/sam3_mask_cache_conf_<c>_qr_filled/ — merged masks (→ step 05)

Already-cached frames are skipped — safe to resume after interruption.

Usage:
  python scripts/04_sam3_mask.py --frames_dir /path/to/frames
"""

import sys
sys.path = [p for p in sys.path if not p.startswith('/usr/local/lib/python3.12')]

import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
import cv2
from multiprocessing import Pool
import torch

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ─────────────────────────────────────────────────────────────────────────────
# SAM3 init + mask caching
# ─────────────────────────────────────────────────────────────────────────────

def initialize_sam3(confidence_threshold=0.1):
    if torch.cuda.is_available():
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("  ✓ Using CUDA with bfloat16")
    else:
        print("  ⚠ CUDA not available, using CPU")
    model     = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
    print("  ✓ SAM3 ready")
    return processor


def cache_masks_frame_multi(image_path, processor, prompts_and_paths):
    """
    Encode the image once, then query each (prompt, cache_path) pair in turn.
    Already-cached paths are skipped; if all are cached the image is never loaded.
    """
    needed = [(p, cp) for p, cp in prompts_and_paths if not os.path.exists(cp)]
    if not needed:
        return

    image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    h, w = image.size[1], image.size[0]

    state = processor.set_image(image)   # encode image ONCE

    for prompt, cache_path in needed:
        # reset_all_prompts clears previous text features + results but keeps
        # the image backbone encoding in state["backbone_out"]
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(state=state, prompt=prompt)

        masks  = state["masks"]
        scores = state["scores"]

        if masks.shape[0] == 0:
            masks_np  = np.zeros((0, h, w), dtype=np.uint8)
            scores_np = np.zeros(0, dtype=np.float32)
        else:
            masks_np  = masks.squeeze(1).cpu().numpy().astype(np.uint8)
            scores_np = scores.float().cpu().numpy()

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, masks=masks_np, scores=scores_np)


# ─────────────────────────────────────────────────────────────────────────────
# QR hole recovery
# ─────────────────────────────────────────────────────────────────────────────

def _recovery_worker(args):
    """
    Per-frame worker (module-level for multiprocessing pickling).
    Returns number of QR masks merged in this frame.
    """
    i, stone_cache_dir, qr_cache_dir, combined_dir, ring_px, min_coverage = args

    out_path = os.path.join(combined_dir, f"masks_{i:06d}.npz")
    if os.path.exists(out_path):
        return 0

    stone_path = os.path.join(stone_cache_dir, f"masks_{i:06d}.npz")
    qr_path    = os.path.join(qr_cache_dir,    f"masks_{i:06d}.npz")

    stone_data   = np.load(stone_path)
    stone_masks  = stone_data['masks'].astype(bool)    # (N, H, W)
    stone_scores = stone_data['scores'].astype(np.float32)

    if not os.path.exists(qr_path):
        np.savez_compressed(out_path, masks=stone_masks.astype(np.uint8),
                            scores=stone_scores)
        return 0

    qr_data  = np.load(qr_path)
    qr_masks = qr_data['masks'].astype(bool)    # (M, H, W)

    if qr_masks.shape[0] == 0 or stone_masks.shape[0] == 0:
        np.savez_compressed(out_path, masks=stone_masks.astype(np.uint8),
                            scores=stone_scores)
        return 0

    # cv2.dilate is ~10× faster than scipy binary_dilation on large kernels
    kernel      = np.ones((ring_px * 2 + 1, ring_px * 2 + 1), dtype=np.uint8)
    merged_masks = stone_masks.copy()
    # precompute float32 view for vectorised coverage sum
    stone_f     = stone_masks.astype(np.float32)   # (N, H, W)
    n_merged    = 0

    for qr_mask in qr_masks:
        expanded  = cv2.dilate(qr_mask.astype(np.uint8), kernel)
        ring      = expanded.astype(bool) & ~qr_mask
        ring_size = int(ring.sum())
        if ring_size == 0:
            continue

        # vectorised: coverage of ring by every stone mask simultaneously
        ring_f    = ring.astype(np.float32)
        coverages = (stone_f * ring_f[np.newaxis]).sum(axis=(1, 2)) / ring_size
        best_idx  = int(coverages.argmax())
        if coverages[best_idx] >= min_coverage:
            merged_masks[best_idx] |= qr_mask
            n_merged += 1

    np.savez_compressed(out_path, masks=merged_masks.astype(np.uint8),
                        scores=stone_scores)
    return n_merged


def apply_qr_recovery(stone_cache_dir, qr_cache_dir, combined_dir, n_frames,
                       ring_px=10, min_coverage=0.5, n_workers=4):
    """
    Parallel QR hole recovery across frames.
    Returns (combined_dir, total_merged_count).
    """
    os.makedirs(combined_dir, exist_ok=True)

    n_done = sum(1 for i in range(n_frames)
                 if os.path.exists(os.path.join(combined_dir, f"masks_{i:06d}.npz")))
    if n_done == n_frames:
        print(f"  QR recovery already complete ({n_done}/{n_frames}) — skipping")
        return combined_dir, 0

    frame_args = [
        (i, stone_cache_dir, qr_cache_dir, combined_dir, ring_px, min_coverage)
        for i in range(n_frames)
    ]

    with Pool(processes=n_workers) as pool:
        results = list(tqdm(pool.imap(_recovery_worker, frame_args),
                            total=n_frames, desc="QR hole recovery"))

    n_merged_total  = sum(results)
    n_frames_merged = sum(1 for r in results if r > 0)
    print(f"  ✓ QR recovery: {n_merged_total} QR regions merged "
          f"across {n_frames_merged} frames → {combined_dir}")
    return combined_dir, n_merged_total


# ─────────────────────────────────────────────────────────────────────────────
# Debug previews
# ─────────────────────────────────────────────────────────────────────────────

def save_debug_previews(color_files, cache_dir, debug_dir, sam_prompt,
                        sam_confidence, n_samples=15):
    """
    Save side-by-side debug images (original | coloured mask overlay) for
    n_samples evenly-spaced frames.  Written to debug_dir/NNNNNN_masks.jpg.

    Each mask is drawn with:
      - semi-transparent colour fill
      - mask ID and confidence score label
    Footer bar on the overlay side shows: prompt and confidence threshold.
    """
    try:
        from PIL import ImageFont
        font    = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = font_sm = ImageFont.load_default()

    os.makedirs(debug_dir, exist_ok=True)
    indices = np.linspace(0, len(color_files) - 1, n_samples, dtype=int)
    rng     = np.random.default_rng(42)

    for idx in indices:
        cache_path = os.path.join(cache_dir, f"masks_{idx:06d}.npz")
        if not os.path.exists(cache_path):
            continue

        data      = np.load(cache_path)
        masks_np  = data['masks']    # (N, H, W) uint8
        scores_np = data['scores']   # (N,) float32

        orig    = Image.open(color_files[idx]).convert('RGB')
        overlay = orig.copy().convert('RGBA')

        colours = [tuple(rng.integers(60, 220, size=3).tolist())
                   for _ in range(len(masks_np))]

        for mask_id, (mask, score, colour) in enumerate(
                zip(masks_np, scores_np, colours)):
            fill_arr            = np.zeros((*mask.shape, 4), dtype=np.uint8)
            fill_arr[mask == 1] = (*colour, 100)
            overlay = Image.alpha_composite(
                overlay, Image.fromarray(fill_arr, 'RGBA'))

            ys, xs = np.where(mask)
            if not len(xs):
                continue
            cx, cy = int(xs.mean()), int(ys.mean())
            label  = f"#{mask_id}  {score:.2f}"
            lw     = len(label) * 7
            ann    = Image.new('RGBA', orig.size, (0, 0, 0, 0))
            d      = ImageDraw.Draw(ann)
            d.rectangle([cx - lw // 2 - 2, cy - 8,
                         cx + lw // 2 + 2, cy + 8], fill=(0, 0, 0, 160))
            d.text((cx - lw // 2, cy - 7), label, fill=(*colour, 255), font=font_sm)
            overlay = Image.alpha_composite(overlay, ann)

        overlay_rgb = overlay.convert('RGB')

        footer_h = 28
        footer   = Image.new('RGB', (orig.width, footer_h), (30, 30, 30))
        fd       = ImageDraw.Draw(footer)
        fd.text((6, 6),
                f'prompt: "{sam_prompt}"   conf: {sam_confidence}   '
                f'masks: {len(masks_np)}   frame: {idx}',
                fill=(220, 220, 220), font=font_sm)

        combined = Image.new('RGB', (orig.width * 2, orig.height + footer_h))
        combined.paste(orig,        (0, 0))
        combined.paste(overlay_rgb, (orig.width, 0))
        combined.paste(footer,      (orig.width, orig.height))

        combined.save(os.path.join(debug_dir, f"{idx:06d}_masks.jpg"), quality=88)

    print(f"  ✓ {len(indices)} debug previews → {debug_dir}/")


def save_qr_recovery_debug(color_files, stone_cache_dir, qr_cache_dir,
                            combined_cache_dir, debug_dir, n_samples=15):
    """
    Three-column debug: original | stone masks | merged masks (QR holes filled).
    Frames where a QR merge happened are highlighted with a red border.
    """
    try:
        from PIL import ImageFont
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font_sm = ImageFont.load_default()

    os.makedirs(debug_dir, exist_ok=True)
    indices = np.linspace(0, len(color_files) - 1, n_samples, dtype=int)
    rng     = np.random.default_rng(7)

    for idx in indices:
        stone_path    = os.path.join(stone_cache_dir,    f"masks_{idx:06d}.npz")
        combined_path = os.path.join(combined_cache_dir, f"masks_{idx:06d}.npz")
        qr_path       = os.path.join(qr_cache_dir,       f"masks_{idx:06d}.npz")
        if not (os.path.exists(stone_path) and os.path.exists(combined_path)):
            continue

        orig = Image.open(color_files[idx]).convert('RGB')
        W, H = orig.size

        def make_overlay(masks_np, tint_colour=None):
            overlay = orig.copy().convert('RGBA')
            colours = [tuple(rng.integers(60, 220, size=3).tolist())
                       for _ in range(len(masks_np))]
            if tint_colour is not None:
                colours = [tint_colour] * len(masks_np)
            for mask, colour in zip(masks_np, colours):
                fill_arr            = np.zeros((*mask.shape, 4), dtype=np.uint8)
                fill_arr[mask == 1] = (*colour, 110)
                overlay = Image.alpha_composite(overlay, Image.fromarray(fill_arr, 'RGBA'))
            return overlay.convert('RGB')

        stone_masks    = np.load(stone_path)['masks']
        combined_masks = np.load(combined_path)['masks']

        qr_masks = np.zeros((0, H, W), dtype=np.uint8)
        if os.path.exists(qr_path):
            qr_masks = np.load(qr_path)['masks']

        stone_ov    = make_overlay(stone_masks)
        combined_ov = make_overlay(combined_masks)

        # Highlight QR masks in red on the combined overlay
        if qr_masks.shape[0] > 0:
            qr_layer = combined_ov.copy().convert('RGBA')
            for qr_mask in qr_masks:
                fill_arr            = np.zeros((*qr_mask.shape, 4), dtype=np.uint8)
                fill_arr[qr_mask == 1] = (220, 30, 30, 160)
                qr_layer = Image.alpha_composite(qr_layer, Image.fromarray(fill_arr, 'RGBA'))
            combined_ov = qr_layer.convert('RGB')

        # Detect if any QR was merged in this frame
        diff = combined_masks.astype(np.int32).sum() - stone_masks.astype(np.int32).sum()
        qr_merged = diff > 0

        footer_h = 28
        total_w  = W * 3

        canvas = Image.new('RGB', (total_w, H + footer_h), (20, 20, 20))
        canvas.paste(orig,        (0, 0))
        canvas.paste(stone_ov,    (W, 0))
        canvas.paste(combined_ov, (W * 2, 0))

        if qr_merged:
            draw = ImageDraw.Draw(canvas)
            draw.rectangle([W * 2, 0, total_w - 1, H - 1], outline=(220, 50, 50), width=4)

        footer = Image.new('RGB', (total_w, footer_h), (30, 30, 30))
        fd     = ImageDraw.Draw(footer)
        qr_txt = f"  QR merged: {qr_masks.shape[0]} detected" if qr_merged else ""
        fd.text((6, 6),
                f"frame {idx}  |  stone masks: {stone_masks.shape[0]}  "
                f"|  combined: {combined_masks.shape[0]}{qr_txt}",
                fill=(220, 220, 220), font=font_sm)
        canvas.paste(footer, (0, H))

        canvas.save(os.path.join(debug_dir, f"{idx:06d}_qr_recovery.jpg"), quality=88)

    print(f"  ✓ {len(indices)} QR recovery previews → {debug_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 04 — SAM3 mask cache (L1) with QR hole recovery")
    parser.add_argument('--frames_dir',       required=True,
                        help='Frames directory (from step 01)')
    parser.add_argument('--sam_prompt',       default="individual stone")
    parser.add_argument('--sam_confidence',   type=float, default=0.1)
    # QR recovery args
    parser.add_argument('--qr_prompt',        default="QR code",
                        help='SAM3 text prompt for QR code detection (default: "QR code")')
    parser.add_argument('--qr_ring_px',       type=int,   default=10,
                        help='Dilation ring size in pixels for enclosure check (default 10)')
    parser.add_argument('--qr_min_coverage',  type=float, default=0.5,
                        help='Minimum fraction of QR ring covered by a stone mask '
                             'to count as enclosed (default 0.5)')
    parser.add_argument('--skip_qr_recovery', action='store_true',
                        help='Skip QR hole recovery; output cache = raw stone cache')
    parser.add_argument('--n_workers',        type=int,   default=4,
                        help='Parallel workers for QR hole recovery (default 4)')
    args = parser.parse_args()

    color_dir = os.path.join(args.frames_dir, 'color')
    color_files = sorted(os.path.join(color_dir, f)
                         for f in os.listdir(color_dir)
                         if f.endswith(('.jpg', '.png')))
    n_total = len(color_files)

    stone_cache_dir = os.path.join(
        args.frames_dir, f'sam3_mask_cache_conf_{args.sam_confidence}')
    qr_cache_dir = os.path.join(
        args.frames_dir, f'sam3_qr_cache_conf_{args.sam_confidence}')
    combined_cache_dir = stone_cache_dir + '_qr_filled'

    n_stone_cached = sum(1 for i in range(n_total)
                         if os.path.exists(
                             os.path.join(stone_cache_dir, f"masks_{i:06d}.npz")))
    n_qr_cached = sum(1 for i in range(n_total)
                      if os.path.exists(
                          os.path.join(qr_cache_dir, f"masks_{i:06d}.npz")))

    print("=" * 60)
    print("Step 04 — SAM3 Mask Cache + QR Hole Recovery")
    print("=" * 60)
    print(f"  frames_dir      : {args.frames_dir}")
    print(f"  stone cache     : {stone_cache_dir}")
    print(f"  qr cache        : {qr_cache_dir}")
    print(f"  combined cache  : {combined_cache_dir}")
    print(f"  frames          : {n_total}  "
          f"(stone cached: {n_stone_cached}, qr cached: {n_qr_cached})")
    print(f"  stone prompt    : {args.sam_prompt!r}")
    print(f"  qr prompt       : {args.qr_prompt!r}")
    print(f"  confidence      : {args.sam_confidence}")
    print(f"  qr_ring_px      : {args.qr_ring_px}")
    print(f"  qr_min_coverage : {args.qr_min_coverage}")

    need_stone = n_stone_cached < n_total
    need_qr    = (not args.skip_qr_recovery) and (n_qr_cached < n_total)

    if need_stone or need_qr:
        print("\nInitializing SAM3…")
        processor = initialize_sam3(args.sam_confidence)
        os.makedirs(stone_cache_dir, exist_ok=True)
        os.makedirs(qr_cache_dir,    exist_ok=True)

        print(f"\nCaching masks (image encoded once per frame; "
              f"stone={need_stone}, qr={need_qr})")
        for i, img_path in enumerate(tqdm(color_files, desc="SAM3 masks")):
            prompts_and_paths = []
            if need_stone:
                prompts_and_paths.append(
                    (args.sam_prompt,
                     os.path.join(stone_cache_dir, f"masks_{i:06d}.npz")))
            if need_qr:
                prompts_and_paths.append(
                    (args.qr_prompt,
                     os.path.join(qr_cache_dir,    f"masks_{i:06d}.npz")))
            cache_masks_frame_multi(img_path, processor, prompts_and_paths)

        if need_stone:
            print(f"  ✓ Stone mask cache → {stone_cache_dir}")
        if need_qr:
            print(f"  ✓ QR mask cache    → {qr_cache_dir}")
    else:
        print("\n  Both mask caches complete — skipping SAM3 inference.")

    # ── QR hole recovery ──────────────────────────────────────────────────────
    if args.skip_qr_recovery:
        output_cache_dir = stone_cache_dir
        print(f"\n  QR recovery skipped — downstream uses: {stone_cache_dir}")
    else:
        print("\nApplying QR hole recovery…")
        output_cache_dir, _ = apply_qr_recovery(
            stone_cache_dir    = stone_cache_dir,
            qr_cache_dir       = qr_cache_dir,
            combined_dir       = combined_cache_dir,
            n_frames           = n_total,
            ring_px            = args.qr_ring_px,
            min_coverage       = args.qr_min_coverage,
            n_workers          = args.n_workers,
        )

    # ── Debug previews ────────────────────────────────────────────────────────
    print("\nSaving debug previews…")
    save_debug_previews(color_files, output_cache_dir,
                        debug_dir=output_cache_dir + "_debug",
                        sam_prompt=args.sam_prompt,
                        sam_confidence=args.sam_confidence)

    if not args.skip_qr_recovery:
        save_qr_recovery_debug(
            color_files,
            stone_cache_dir    = stone_cache_dir,
            qr_cache_dir       = qr_cache_dir,
            combined_cache_dir = combined_cache_dir,
            debug_dir          = combined_cache_dir + "_debug",
        )

    print(f"\n✓ Step 04 complete")
    print(f"  Pass 1 (stone) : {stone_cache_dir}/")
    print(f"  Pass 2 (QR)    : {qr_cache_dir}/")
    if not args.skip_qr_recovery:
        print(f"  Combined (→ 05): {combined_cache_dir}/")
        print(f"\n  Set MASK_CACHE_DIR={combined_cache_dir} in 05_sam3_score.sh")
    else:
        print(f"\n  Set MASK_CACHE_DIR={stone_cache_dir} in 05_sam3_score.sh")


if __name__ == "__main__":
    main()
