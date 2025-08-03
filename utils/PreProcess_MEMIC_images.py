#!/usr/bin/env python3
"""
PreProcess_MEMIC_images.py

Provides a function to preprocess MEMIC Z-stack TIFF files:
groups raw TIFF slices by domain and writes out a multi-page TIFF
stack named by domain into the matching subfolder of the output root.
"""

import os
import glob
import tifffile
import numpy as np

# The canonical imaging order
ORDER = [
    "f02", "f03", "f04", "f05", "f06", "f07", "f08", "f09", "f10",
    "f19", "f18", "f17", "f16", "f15", "f14", "f13", "f12", "f11",
    "f20", "f21", "f22", "f23", "f01", "f24", "f25", "f26", "f27",
    "f36", "f35", "f34", "f33", "f32", "f31", "f30", "f29", "f28",
    "f37", "f38", "f39", "f40", "f41", "f42", "f43", "f44", "f45"
]

def preprocess_one_tp(input_root: str, output_root: str, well: str, tp: str):
    """
    Process a single timepoint folder: group raw TIFFs by domain and
    write BF/FL stacks under output_root/well/tp.
    """
    tp_path = os.path.join(input_root, well, tp)
    if not os.path.isdir(tp_path):
        return
    out_tp = os.path.join(output_root, well, tp)
    os.makedirs(out_tp, exist_ok=True)

    # Gather all TIFF files in this time-point folder
    tifs = glob.glob(os.path.join(tp_path, "*.tif*"))

    # Group by domain ID
    groups = {}
    for path in tifs:
        name = os.path.basename(path)
        if len(name) < 9:
            continue
        domain = name[6:9]
        groups.setdefault(domain, []).append(path)

    # Process each domain in canonical ORDER
    for domain in ORDER:
        # Split domain files into BF and FL
        bf_files = [p for p in groups.get(domain, []) if '-ch2' in os.path.basename(p)]
        fl_files = [p for p in groups.get(domain, []) if '-ch1' in os.path.basename(p)]

        def z_index(p):
            bn = os.path.basename(p)
            try:
                return int(bn[9:12])
            except ValueError:
                return 0

        # BF stack
        if bf_files:
            bf_files = sorted(bf_files, key=z_index)
            bf_stack_list = [tifffile.imread(p) for p in bf_files]
            bf_stack_list = [arr.astype(np.float32) if arr.dtype.kind=='f' else arr for arr in bf_stack_list]
            bf_stack = np.stack(bf_stack_list, axis=0)
            bf_out = os.path.join(out_tp, f"{domain}_bf.tiff")
            tifffile.imwrite(bf_out, bf_stack, imagej=True)

        # FL stack
        if fl_files:
            fl_files = sorted(fl_files, key=z_index)
            fl_stack_list = [tifffile.imread(p) for p in fl_files]
            fl_stack_list = [arr.astype(np.float32) if arr.dtype.kind=='f' else arr for arr in fl_stack_list]
            fl_stack = np.stack(fl_stack_list, axis=0)
            fl_out = os.path.join(out_tp, f"{domain}_fl.tiff")
            tifffile.imwrite(fl_out, fl_stack, imagej=True)

def preprocess_memic_images(input_root: str, output_root: str):
    """
    Walk through wells and time-points, parallel-ready.
    """
    for well in sorted(os.listdir(input_root)):
        well_path = os.path.join(input_root, well)
        if not os.path.isdir(well_path):
            continue
        for tp in sorted(os.listdir(well_path)):
            preprocess_one_tp(input_root, output_root, well, tp)