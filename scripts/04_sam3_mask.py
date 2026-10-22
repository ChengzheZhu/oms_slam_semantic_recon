#!/usr/bin/env python3
"""
04_sam3_mask.py — Pre-compute SAM3 raw mask cache for all extracted frames (L1 cache).

Single-pass approach (image encoded once per frame, two prompts share the encoding):
  Prompt 1: --sam_prompt   ("individual stone") → stone masks
  Prompt 2: --label_prompt ("paper label")      → label masks

Label recovery (CPU post-process) → merged cache read by step 05:
  Paper labels on stones read as holes or notches in the stone mask. Each
  label is grown by --label_ring_px; the outer edge of the grown mask is its
  perimeter. The stone covering the largest share of that perimeter takes the
  label if the share is >= --label_min_coverage. The merged area is the label
  grown by --label_grow_px, minus pixels of other stones.

Writes (names from sam3_cache.py; raw caches are named by prompt):
  <frames_dir>/sam3_<stone prompt>_conf_<c>/                — raw stone masks (L1)
  <frames_dir>/sam3_<label prompt>_conf_<c>/                — raw label masks
  <frames_dir>/sam3_merged_<stone>+<label>_conf_<c>/        — merged masks (→ step 05)
      masks_NNNNNN.npz: masks, scores; params.json: label recovery settings

Already-cached frames are skipped — safe to resume after interruption. The
merged cache is rebuilt when its params.json no longer matches the settings.
With --skip_label_recovery, step 05 reads the raw stone cache.

Usage:
  python scripts/04_sam3_mask.py --frames_dir /path/to/frames
"""

import sys
sys.path = [p for p in sys.path if not p.startswith('/usr/local/lib/python3.12')]

import os
import argparse
import glob
import json
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
import cv2
from multiprocessing import Pool
import torch

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from sam3_cache import merged_dir, raw_dir


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
# Label recovery
# ─────────────────────────────────────────────────────────────────────────────

def _disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _perimeter(mask, ring_px):
    """(ys, xs) of the outer edge of mask grown by ring_px, minus image-edge pixels."""
    grown = cv2.dilate(mask.astype(np.uint8), _disk(ring_px))
    contours, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros(0, int), np.zeros(0, int)
    pts = np.unique(np.vstack(contours)[:, 0], axis=0)      # (x, y)
    h, w = mask.shape
    # where the grown label is clipped by the frame the edge is not a real perimeter
    inside = (pts[:, 0] > 0) & (pts[:, 0] < w - 1) & (pts[:, 1] > 0) & (pts[:, 1] < h - 1)
    return pts[inside, 1], pts[inside, 0]


def merge_labels_frame(stone_masks, stone_scores, label_masks,
                       ring_px, min_coverage, grow_px):
    """Label recovery for one frame (see module docstring). Returns (masks, stats)."""
    merged = stone_masks.copy()
    n_merged = 0
    if len(stone_masks) and len(label_masks):
        claims = stone_masks.sum(axis=0, dtype=np.int32)
        grow_kernel = _disk(grow_px) if grow_px > 0 else None
        for label in label_masks:
            ys, xs = _perimeter(label, ring_px)
            if not len(ys):
                continue
            coverage = stone_masks[:, ys, xs].mean(axis=1)    # perimeter share per stone
            best = int(coverage.argmax())
            if coverage[best] < min_coverage:
                continue
            grown = (cv2.dilate(label.astype(np.uint8), grow_kernel).astype(bool)
                     if grow_kernel is not None else label)
            other_stones = (claims - stone_masks[best]) > 0
            merged[best] |= grown & ~other_stones
            n_merged += 1
    return merged, dict(labels=len(label_masks), merged=n_merged)


def _recovery_worker(args):
    """Per-frame worker (module-level for multiprocessing pickling). Returns stats or None."""
    (i, stone_cache_dir, label_cache_dir, merged_cache_dir,
     ring_px, min_coverage, grow_px) = args

    out_path = os.path.join(merged_cache_dir, f"masks_{i:06d}.npz")
    if os.path.exists(out_path):
        return None

    stone_data   = np.load(os.path.join(stone_cache_dir, f"masks_{i:06d}.npz"))
    stone_masks  = stone_data['masks'].astype(bool)          # (N, H, W)
    stone_scores = stone_data['scores'].astype(np.float32)
    label_masks  = np.load(os.path.join(label_cache_dir, f"masks_{i:06d}.npz")
                           )['masks'].astype(bool)           # (M, H, W)

    masks, stats = merge_labels_frame(stone_masks, stone_scores, label_masks,
                                      ring_px, min_coverage, grow_px)
    np.savez_compressed(out_path, masks=masks.astype(np.uint8), scores=stone_scores)
    return stats


def apply_label_recovery(stone_cache_dir, label_cache_dir, merged_cache_dir, n_frames,
                         ring_px=3, min_coverage=0.6, grow_px=2, n_workers=4):
    """
    Parallel label recovery across frames. Frames already in merged_cache_dir
    are kept only if its params.json matches these settings.
    """
    os.makedirs(merged_cache_dir, exist_ok=True)
    params = dict(stone_cache=os.path.basename(stone_cache_dir),
                  label_cache=os.path.basename(label_cache_dir),
                  label_ring_px=ring_px,
                  label_min_coverage=min_coverage, label_grow_px=grow_px)
    params_path = os.path.join(merged_cache_dir, 'params.json')
    old_params = None
    if os.path.exists(params_path):
        with open(params_path) as f:
            old_params = json.load(f)
    stale = sorted(glob.glob(os.path.join(merged_cache_dir, 'masks_*.npz')))
    if stale and old_params != params:
        print(f"  Merged cache settings changed ({old_params} → {params}); "
              f"rebuilding {len(stale)} frames")
        for path in stale:
            os.remove(path)
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)

    n_done = sum(1 for i in range(n_frames)
                 if os.path.exists(os.path.join(merged_cache_dir, f"masks_{i:06d}.npz")))
    if n_done == n_frames:
        print(f"  Label recovery already complete ({n_done}/{n_frames}) — skipping")
        return merged_cache_dir

    frame_args = [
        (i, stone_cache_dir, label_cache_dir, merged_cache_dir,
         ring_px, min_coverage, grow_px)
        for i in range(n_frames)
    ]
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(pool.imap(_recovery_worker, frame_args),
                            total=n_frames, desc="Label recovery"))

    total = {}
    for stats in results:
        for k, v in (stats or {}).items():
            total[k] = total.get(k, 0) + v
    if total:
        print(f"  ✓ Labels merged: {total['merged']}/{total['labels']}")
    print(f"  → {merged_cache_dir}")
    return merged_cache_dir


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



def save_label_recovery_debug(color_files, stone_cache_dir, label_cache_dir,
                              merged_cache_dir, debug_dir, n_samples=15):
    """
    Three-column debug: original | raw stone masks | merged masks
    (label masks outlined red).
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
        stone_path  = os.path.join(stone_cache_dir,  f"masks_{idx:06d}.npz")
        merged_path = os.path.join(merged_cache_dir, f"masks_{idx:06d}.npz")
        if not (os.path.exists(stone_path) and os.path.exists(merged_path)):
            continue

        orig = Image.open(color_files[idx]).convert('RGB')
        W, H = orig.size

        def make_overlay(masks_np):
            overlay = orig.copy().convert('RGBA')
            for mask in masks_np:
                colour              = tuple(rng.integers(60, 220, size=3).tolist())
                fill_arr            = np.zeros((*mask.shape, 4), dtype=np.uint8)
                fill_arr[mask == 1] = (*colour, 110)
                overlay = Image.alpha_composite(overlay, Image.fromarray(fill_arr, 'RGBA'))
            return np.asarray(overlay.convert('RGB')).copy()

        stone_masks = np.load(stone_path)['masks']
        merged      = np.load(merged_path)
        label_masks = np.zeros((0, H, W), dtype=np.uint8)
        label_path  = os.path.join(label_cache_dir, f"masks_{idx:06d}.npz")
        if os.path.exists(label_path):
            label_masks = np.load(label_path)['masks']

        merged_ov = make_overlay(merged['masks'])
        for label in label_masks:
            contours, _ = cv2.findContours(label.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(merged_ov, contours, -1, (230, 30, 30), 1)

        footer_h = 28
        total_w  = W * 3
        canvas = Image.new('RGB', (total_w, H + footer_h), (20, 20, 20))
        canvas.paste(orig,                                       (0, 0))
        canvas.paste(Image.fromarray(make_overlay(stone_masks)), (W, 0))
        canvas.paste(Image.fromarray(merged_ov),                 (W * 2, 0))

        footer = Image.new('RGB', (total_w, footer_h), (30, 30, 30))
        fd     = ImageDraw.Draw(footer)
        fd.text((6, 6),
                f"frame {idx}  |  stone masks: {stone_masks.shape[0]}  "
                f"|  labels: {label_masks.shape[0]} (red)",
                fill=(220, 220, 220), font=font_sm)
        canvas.paste(footer, (0, H))

        canvas.save(os.path.join(debug_dir, f"{idx:06d}_label_recovery.jpg"), quality=88)

    print(f"  ✓ {len(indices)} label recovery previews → {debug_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 04 — SAM3 mask cache (L1) with label recovery")
    parser.add_argument('--frames_dir',       required=True,
                        help='Frames directory (from step 01)')
    parser.add_argument('--sam_prompt',       default="individual stone")
    parser.add_argument('--sam_confidence',   type=float, default=0.1)
    # Label recovery args
    parser.add_argument('--label_prompt',     default="paper label",
                        help='SAM3 text prompt for the labels on stones (default: "paper label")')
    parser.add_argument('--label_ring_px',    type=int,   default=3,
                        help='Label grown by this many px; the grown outer edge is the '
                             'perimeter checked against stone masks (default 3)')
    parser.add_argument('--label_min_coverage', type=float, default=0.6,
                        help='Share of that perimeter a stone mask must cover to take '
                             'the label (default 0.6)')
    parser.add_argument('--label_grow_px',    type=int,   default=2,
                        help='Merged area = label grown by this many px, never into '
                             'another stone (default 2)')
    parser.add_argument('--skip_label_recovery', action='store_true',
                        help='No label prompt or merge; step 05 reads the raw stone cache')
    parser.add_argument('--n_workers',        type=int,   default=4,
                        help='Parallel workers for label recovery (default 4)')
    args = parser.parse_args()

    color_dir = os.path.join(args.frames_dir, 'color')
    color_files = sorted(os.path.join(color_dir, f)
                         for f in os.listdir(color_dir)
                         if f.endswith(('.jpg', '.png')))
    n_total = len(color_files)

    stone_cache_dir = raw_dir(args.frames_dir, args.sam_prompt, args.sam_confidence)
    label_cache_dir = (None if args.skip_label_recovery else
                       raw_dir(args.frames_dir, args.label_prompt, args.sam_confidence))
    merged_cache_dir = (None if args.skip_label_recovery else
                        merged_dir(args.frames_dir, args.sam_confidence,
                                   args.sam_prompt, args.label_prompt))

    def n_cached(cache_dir):
        return sum(1 for i in range(n_total)
                   if os.path.exists(os.path.join(cache_dir, f"masks_{i:06d}.npz")))
    n_stone_cached = n_cached(stone_cache_dir)
    n_label_cached = n_cached(label_cache_dir) if label_cache_dir else 0

    print("=" * 60)
    print("Step 04 — SAM3 Mask Cache + Label Recovery")
    print("=" * 60)
    print(f"  frames_dir      : {args.frames_dir}")
    print(f"  stone cache     : {stone_cache_dir}")
    print(f"  label cache     : {label_cache_dir or '(label recovery skipped)'}")
    print(f"  merged cache    : {merged_cache_dir or '(label recovery skipped)'}")
    print(f"  frames          : {n_total}  "
          f"(stone cached: {n_stone_cached}, label cached: {n_label_cached})")
    print(f"  stone prompt    : {args.sam_prompt!r}")
    print(f"  label prompt    : {args.label_prompt!r}")
    print(f"  confidence      : {args.sam_confidence}")
    print(f"  label ring/cov/grow : {args.label_ring_px} px / "
          f"{args.label_min_coverage} / {args.label_grow_px} px")

    # Caches from before prompt-named dirs: the prompt was not recorded.
    legacy = os.path.join(args.frames_dir,
                          f'sam3_mask_cache_conf_{float(args.sam_confidence)}')
    if n_stone_cached == 0 and os.path.isdir(legacy):
        print(f"\n  NOTE: found {legacy} (older naming, prompt not recorded).\n"
              f"  If it was made with {args.sam_prompt!r}, rename it to reuse it:\n"
              f"    mv {legacy} {stone_cache_dir}")

    need_stone = n_stone_cached < n_total
    need_label = label_cache_dir is not None and n_label_cached < n_total

    if need_stone or need_label:
        print("\nInitializing SAM3…")
        processor = initialize_sam3(args.sam_confidence)

        print(f"\nCaching masks (image encoded once per frame; "
              f"stone={need_stone}, label={need_label})")
        for i, img_path in enumerate(tqdm(color_files, desc="SAM3 masks")):
            prompts_and_paths = []
            if need_stone:
                prompts_and_paths.append(
                    (args.sam_prompt,
                     os.path.join(stone_cache_dir, f"masks_{i:06d}.npz")))
            if need_label:
                prompts_and_paths.append(
                    (args.label_prompt,
                     os.path.join(label_cache_dir, f"masks_{i:06d}.npz")))
            cache_masks_frame_multi(img_path, processor, prompts_and_paths)

        if need_stone:
            print(f"  ✓ Stone mask cache → {stone_cache_dir}")
        if need_label:
            print(f"  ✓ Label mask cache → {label_cache_dir}")
    else:
        print("\n  Mask caches complete — skipping SAM3 inference.")

    # ── Label recovery ────────────────────────────────────────────────────────
    if args.skip_label_recovery:
        output_cache_dir = stone_cache_dir
        print(f"\n  Label recovery skipped — downstream uses: {stone_cache_dir}")
    else:
        print("\nApplying label recovery…")
        output_cache_dir = apply_label_recovery(
            stone_cache_dir  = stone_cache_dir,
            label_cache_dir  = label_cache_dir,
            merged_cache_dir = merged_cache_dir,
            n_frames         = n_total,
            ring_px          = args.label_ring_px,
            min_coverage     = args.label_min_coverage,
            grow_px          = args.label_grow_px,
            n_workers        = args.n_workers,
        )

    # ── Debug previews ────────────────────────────────────────────────────────
    print("\nSaving debug previews…")
    save_debug_previews(color_files, output_cache_dir,
                        debug_dir=output_cache_dir + "_debug",
                        sam_prompt=args.sam_prompt,
                        sam_confidence=args.sam_confidence)
    if not args.skip_label_recovery:
        save_label_recovery_debug(
            color_files,
            stone_cache_dir  = stone_cache_dir,
            label_cache_dir  = label_cache_dir,
            merged_cache_dir = merged_cache_dir,
            debug_dir        = merged_cache_dir + "_debug",
        )

    print(f"\n✓ Step 04 complete")
    print(f"  Pass 1 (stone) : {stone_cache_dir}/")
    if label_cache_dir:
        print(f"  Pass 2 (label) : {label_cache_dir}/")
    print(f"  → step 05      : {output_cache_dir}/")


if __name__ == "__main__":
    main()
