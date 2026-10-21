#!/usr/bin/env python3
"""
config_to_env.py — Flatten pipeline.yaml into bash assignments.

section.key → SECTION_KEY. Each assignment only fires if the variable is not
already set, so values set in a step .sh (or the environment) win.

Usage (from scripts/pipeline_env.sh):
  eval "$(python scripts/config_to_env.py pipeline.yaml)"
"""

import shlex
import sys

import yaml


def to_shell(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if value is None:
        return ''
    if isinstance(value, list):
        return ' '.join(to_shell(v) for v in value)
    return str(value)


def main():
    with open(sys.argv[1]) as f:
        cfg = yaml.safe_load(f) or {}
    for section, entries in cfg.items():
        if not isinstance(entries, dict):
            sys.exit(f"config_to_env: top-level '{section}' must be a section")
        for key, value in entries.items():
            if isinstance(value, dict):
                sys.exit(f"config_to_env: '{section}.{key}' nests too deep")
            name = f"{section}_{key}".upper()
            print(f'if [ -z "${{{name}+x}}" ]; then '
                  f'{name}={shlex.quote(to_shell(value))}; fi')


if __name__ == '__main__':
    main()
