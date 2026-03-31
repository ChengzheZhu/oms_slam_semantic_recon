#!/usr/bin/env python3
"""
Debug SAM3 segmentation by visualizing masks on sample frames.
"""

import sys
sys.path.insert(0, '/home/chengzhe/projects/sam3')

import torch
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
import os
from pathlib import Path

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def visualize_sam3_results(image_path, processor, prompt, confidence, max_size_ratio, output_path):
    """Visualize SAM3 segmentation on a single image."""

    # Load image
    image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Run SAM3
    inference_state = processor.set_image(image)
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)

    masks = inference_state["masks"]
    scores = inference_state["scores"]

    # Filter by size
    img_area = image.size[0] * image.size[1]
    mask_areas = [mask.sum().item() / img_area for mask in masks]
    valid_indices = [i for i, area in enumerate(mask_areas) if area <= max_size_ratio]

    masks = [masks[i] for i in valid_indices]
    scores = [scores[i] for i in valid_indices]
    mask_areas = [mask_areas[i] for i in valid_indices]

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # All masks overlaid
    axes[0, 1].imshow(image)
    for i, mask in enumerate(masks):
        mask_np = mask.squeeze(0).cpu().numpy()
        color = np.random.random(3)

        # Create colored overlay
        colored_mask = np.zeros((*mask_np.shape, 3))
        colored_mask[mask_np] = color

        axes[0, 1].imshow(colored_mask, alpha=0.5)

    axes[0, 1].set_title(f'All Masks ({len(masks)} detected)')
    axes[0, 1].axis('off')

    # Mask boundaries
    axes[1, 0].imshow(image)
    for i, mask in enumerate(masks):
        mask_np = mask.squeeze(0).cpu().numpy().astype(np.uint8)

        # Find contours
        import cv2
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            contour = contour.squeeze()
            if len(contour) > 2:
                axes[1, 0].plot(contour[:, 0], contour[:, 1], linewidth=2)

    axes[1, 0].set_title('Mask Boundaries')
    axes[1, 0].axis('off')

    # Statistics
    axes[1, 1].axis('off')
    stats_text = f"""
SAM3 Detection Statistics

Prompt: "{prompt}"
Confidence: {confidence}
Max Size Ratio: {max_size_ratio}

Total Detections: {len(inference_state["masks"])}
After Size Filter: {len(masks)}

Mask Details:
"""

    for i, (score, area) in enumerate(zip(scores[:10], mask_areas[:10])):  # Show first 10
        stats_text += f"\n  Mask {i+1}: score={score:.3f}, area={area*100:.1f}%"

    if len(masks) > 10:
        stats_text += f"\n  ... and {len(masks)-10} more"

    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return len(masks)


def main():
    parser = argparse.ArgumentParser(description="Debug SAM3 segmentation")
    parser.add_argument('--frames_dir', type=str, required=True,
                       help='Directory with color/ subdirectory')
    parser.add_argument('--output_dir', type=str, default='debug_sam3_output',
                       help='Output directory for visualizations')
    parser.add_argument('--num_frames', type=int, default=10,
                       help='Number of frames to visualize')
    parser.add_argument('--frame_skip', type=int, default=200,
                       help='Frame skip (sample every Nth frame)')
    parser.add_argument('--sam_prompt', type=str, default="individual stone",
                       help='SAM3 text prompt')
    parser.add_argument('--sam_confidence', type=float, default=0.1,
                       help='SAM3 confidence threshold')
    parser.add_argument('--sam_max_size_ratio', type=float, default=0.15,
                       help='Max segment size ratio')

    args = parser.parse_args()

    print("="*80)
    print("SAM3 Mask Debugging Tool")
    print("="*80)

    # Initialize SAM3
    print("\nInitializing SAM3...")
    if torch.cuda.is_available():
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("  ✓ Using CUDA with bfloat16")

    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=args.sam_confidence)
    print("  ✓ SAM3 ready")

    # Get frame list
    color_dir = os.path.join(args.frames_dir, 'color')
    frame_files = sorted([f for f in os.listdir(color_dir) if f.endswith(('.png', '.jpg'))])

    print(f"\nFound {len(frame_files)} frames")

    # Sample frames
    sample_indices = list(range(0, len(frame_files), args.frame_skip))[:args.num_frames]

    print(f"Sampling {len(sample_indices)} frames: {sample_indices}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each frame
    print(f"\nProcessing frames...")
    total_detections = []

    for i, idx in enumerate(sample_indices):
        frame_file = frame_files[idx]
        frame_path = os.path.join(color_dir, frame_file)
        output_path = os.path.join(args.output_dir, f"debug_frame_{idx:06d}.png")

        print(f"\n  Frame {i+1}/{len(sample_indices)}: {frame_file}")

        num_masks = visualize_sam3_results(
            frame_path, processor, args.sam_prompt,
            args.sam_confidence, args.sam_max_size_ratio,
            output_path
        )

        total_detections.append(num_masks)
        print(f"    Detected: {num_masks} segments")
        print(f"    Saved: {output_path}")

    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"\nFrames processed: {len(sample_indices)}")
    print(f"Total detections: {sum(total_detections)}")
    print(f"Average per frame: {sum(total_detections)/len(total_detections):.1f}")
    print(f"Min/Max: {min(total_detections)} / {max(total_detections)}")

    print(f"\nVisualization saved to: {args.output_dir}/")
    print("\nCheck the images to verify:")
    print("  1. Are stones being detected?")
    print("  2. Are detections accurate?")
    print("  3. Is the prompt working correctly?")
    print("  4. Should confidence or size thresholds be adjusted?")

    if sum(total_detections) == 0:
        print("\n⚠ WARNING: No segments detected in any frame!")
        print("  Try:")
        print("    - Lower --sam_confidence (e.g., 0.05)")
        print("    - Different --sam_prompt (e.g., 'stone', 'rock', 'brick')")
        print("    - Higher --sam_max_size_ratio (e.g., 0.3)")
    elif sum(total_detections) / len(total_detections) < 2:
        print("\n⚠ WARNING: Very few detections per frame")
        print("  Consider adjusting parameters as above")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
