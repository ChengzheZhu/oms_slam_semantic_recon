# RealSense Bag File Trimming Guide

## Quick Start

Select the best portion of your .bag file for reconstruction:

```bash
cd /home/chengzhe/projects/_OMS/ORB_SLAM3_RGBD_DenseSlamReconstrction

# Interactive mode (recommended)
python scripts/trim_bag.py --bag /path/to/your/file.bag --preview

# Manual mode
python scripts/trim_bag.py \
  --bag /path/to/your/file.bag \
  --start 100 \
  --end 1500 \
  --output /path/to/output_trimmed.bag
```

## Usage Modes

### Mode 1: Interactive Preview (Recommended)

```bash
python scripts/trim_bag.py \
  --bag /home/chengzhe/Data/OMS_data3/rs_bags/1101/20251101_235516.bag \
  --preview
```

**What it does:**
1. Shows bag file info (total frames, duration, FPS)
2. Opens OpenCV preview window
3. You navigate and mark frames:
   - Press `s` to save frame markers
   - Press `SPACE` to pause/resume
   - Press `q` to finish
4. Uses first and last markers as start/end
5. Confirms and trims

**Controls:**
- `s` - Save current frame number
- `SPACE` - Pause/resume playback
- `q` - Quit preview and proceed

### Mode 2: Manual Frame Selection

```bash
python scripts/trim_bag.py \
  --bag /home/chengzhe/Data/OMS_data3/rs_bags/1101/20251101_235516.bag \
  --start 500 \
  --end 2000
```

Directly specify start and end frames.

### Mode 3: Prompt Mode

```bash
python scripts/trim_bag.py \
  --bag /home/chengzhe/Data/OMS_data3/rs_bags/1101/20251101_235516.bag
```

Script will prompt you for start and end frames.

## Complete Workflow

### Step 1: Trim Bag File

```bash
# Using interactive preview
python scripts/trim_bag.py \
  --bag /home/chengzhe/Data/OMS_data3/rs_bags/1101/20251101_235516.bag \
  --preview \
  --skip 10
```

This creates: `20251101_235516_trimmed_START_END.bag`

### Step 2: Update Pipeline Script

Edit `run_quick_test_pipeline.sh`:

```bash
# Change this line:
BAG_FILE="/home/chengzhe/Data/OMS_data3/rs_bags/1101/20251101_235516.bag"

# To your trimmed file:
BAG_FILE="/home/chengzhe/Data/OMS_data3/rs_bags/1101/20251101_235516_trimmed_500_2000.bag"
```

### Step 3: Run Full Pipeline

```bash
bash run_quick_test_pipeline.sh
```

This will:
1. Extract all frames from trimmed bag
2. Run ORB-SLAM3 for tracking
3. Convert trajectory
4. Run SAM3 boundary reconstruction

## Tips for Selecting Good Segments

### What to Look For

✓ **Smooth camera motion** - steady scanning without jerky movements
✓ **Good coverage** - wall area is fully visible
✓ **Consistent lighting** - no sudden brightness changes
✓ **Clear features** - stones are in focus, not blurry
✓ **Overlapping views** - same area seen from multiple angles

### What to Avoid

✗ **Motion blur** - moving too fast
✗ **Poor lighting** - too dark or overexposed
✗ **Occlusions** - objects blocking view
✗ **Start/end transitions** - first few frames often unstable
✗ **Loop closures issues** - if you return to start, ORB-SLAM3 may drift

### Recommended Segment Length

- **Minimum**: 500 frames (~15 seconds at 30fps)
- **Optimal**: 1000-2000 frames (~30-60 seconds)
- **Maximum**: Depends on computer memory (4000+ frames may be slow)

For best results:
- Trim out initialization (first ~50 frames)
- Trim out ending (last ~50 frames)
- Keep steady scanning portion only

## Example Session

```bash
$ python scripts/trim_bag.py --bag 20251101_235516.bag --preview

================================================================================
RealSense Bag File Trimmer
================================================================================

Analyzing bag file: 20251101_235516.bag

Bag Info:
  Total frames: 4128
  Duration: 137.60 seconds
  FPS: 30.00

Previewing frames 0 to end
Press 'q' to quit preview, 's' to save current frame number
Press SPACE to pause/resume

  Saved frame marker: 523
  Saved frame marker: 2156

Saved frame markers: [523, 2156]
Use frames 523 to 2156? (y/n): y

Trim Summary:
  Frames: 523 to 2156 (1634 frames)
  Duration: ~54.47 seconds
  Output: 20251101_235516_trimmed_523_2156.bag

Proceed with trimming? (y/n): y

Trimming bag file...
  Input: 20251101_235516.bag
  Output: 20251101_235516_trimmed_523_2156.bag
  Frames: 523 to 2156
  Processed 1634 frames...
✓ Trimmed bag file saved
```

## Advanced Options

### Custom Preview Skip

Preview every Nth frame (faster for long bags):

```bash
python scripts/trim_bag.py \
  --bag file.bag \
  --preview \
  --skip 20  # Show every 20th frame
```

### Specify Output Path

```bash
python scripts/trim_bag.py \
  --bag input.bag \
  --start 500 \
  --end 2000 \
  --output /custom/path/output.bag
```

## Troubleshooting

### Preview window doesn't open

Make sure you're in the `sam3_open3d` environment:
```bash
conda activate sam3_open3d
```

### "rs-convert not available"

Normal - script will use Python API (slightly slower but works fine)

### Trimming takes long time

For very long segments, use smaller ranges or consider:
- Using `--skip 1` for preview (show every frame)
- Checking disk space for output file

## Integration with Pipeline

After trimming, update your pipeline configuration:

```bash
# Option 1: Edit run_quick_test_pipeline.sh
vim run_quick_test_pipeline.sh
# Change BAG_FILE path

# Option 2: Create custom pipeline script
cp run_quick_test_pipeline.sh run_custom_pipeline.sh
# Edit BAG_FILE in the new script

# Run
bash run_custom_pipeline.sh
```

The pipeline will handle everything automatically from the trimmed bag.
