#!/usr/bin/env python3
"""
run_preprocess_memic.py

Simple CLI wrapper for PreProcess_MEMIC_images.preprocess_memic_images.
"""

import sys
import os
# Add utils directory to Python path so we can import from it
script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir  = os.path.join(script_dir, 'utils')
sys.path.insert(0, utils_dir)

import concurrent.futures
from PreProcess_MEMIC_images import preprocess_one_tp

def main():
    if len(sys.argv) != 3:
        print("Usage: run_preprocess_memic.py <input_root> <output_root>")
        sys.exit(1)
    input_root = sys.argv[1]
    output_root = sys.argv[2]

    print(f"Preprocessing MEMIC images in parallel:\n  input:  {input_root}\n  output: {output_root}")

    # Collect all (well, tp) pairs
    tasks = []
    for well in sorted(os.listdir(input_root)):
        well_path = os.path.join(input_root, well)
        if not os.path.isdir(well_path):
            continue
        for tp in sorted(os.listdir(well_path)):
            tasks.append((input_root, output_root, well, tp))

    # Run each tp on its own core (reserve 1 core for system)
    max_workers = max(1, os.cpu_count() - 1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as exec:
        futures = [exec.submit(preprocess_one_tp, *args) for args in tasks]
        for fut in concurrent.futures.as_completed(futures):
            # propagate exceptions if any
            fut.result()

    print("Done.")

if __name__ == "__main__":
    main()