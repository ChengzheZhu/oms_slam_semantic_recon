#!/usr/bin/env python3
"""
sam3_cache.py — SAM3 mask-cache directory names (step 04; step 05's input is
derived from the same names in pipeline_env.sh).

Raw caches are named by their prompt, so changing a prompt never reuses masks
made for another one. The merged cache (step 04 label recovery → step 05) is
named by both prompts; its other settings are checked via params.json.

  <frames_dir>/sam3_<prompt>_conf_<c>/                raw masks for one prompt
  <frames_dir>/sam3_merged_<stone>+<label>_conf_<c>/  stone masks, labels merged

Usage (prints step 05's mask cache dir: the merged cache, or the raw stone
cache when the label prompt is omitted, i.e. label recovery is skipped):
  python scripts/sam3_cache.py <frames_dir> <confidence> <stone_prompt> [<label_prompt>]
"""

import os
import re
import sys


def slug(prompt):
    return re.sub(r'[^a-z0-9]+', '_', prompt.lower()).strip('_')


def conf_str(confidence):
    return str(float(confidence))


def raw_dir(frames_dir, prompt, confidence):
    return os.path.join(frames_dir, f'sam3_{slug(prompt)}_conf_{conf_str(confidence)}')


def merged_dir(frames_dir, confidence, stone_prompt, label_prompt):
    name = f'{slug(stone_prompt)}+{slug(label_prompt)}'
    return os.path.join(frames_dir, f'sam3_merged_{name}_conf_{conf_str(confidence)}')


if __name__ == '__main__':
    if len(sys.argv) not in (4, 5):
        sys.exit(__doc__)
    frames_dir, confidence, stone_prompt = sys.argv[1:4]
    label_prompt = sys.argv[4] if len(sys.argv) == 5 and sys.argv[4] else None
    print(merged_dir(frames_dir, confidence, stone_prompt, label_prompt) if label_prompt
          else raw_dir(frames_dir, stone_prompt, confidence))
