#!/usr/bin/env bash
set -euo pipefail
python run_comparison.py --seeds 50 --start-seed 101 --months 120 --patients 100 --insurers 4 --output .
