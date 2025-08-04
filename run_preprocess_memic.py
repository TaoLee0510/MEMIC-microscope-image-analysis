#!/usr/bin/env python3
# run_pipeline.py

import sys
import os
from stitch     import stitch_images
from crop       import crop_images
from segment    import segment_tiles
from analysis   import analyze_counts

def main():
    if len(sys.argv)!=3:
        print("Usage: run_pipeline.py <EDoF_root> <Output_root>")
        sys.exit(1)

    edof_root   = sys.argv[1].rstrip(os.sep)
    out_root    = sys.argv[2].rstrip(os.sep)

    # define sub-roots for each step
    stitched_root  = os.path.join(out_root, "stitched")
    cropped_root   = os.path.join(out_root, "cropped")
    segmented_root = os.path.join(out_root, "segmented")
    analysis_root  = os.path.join(out_root, "analysis")

    print("=== Step 2: stitching ===")
    stitch_images(edof_root, stitched_root)

    print("\n=== Step 3: cropping ===")
    crop_images(stitched_root, cropped_root)

    print("\n=== Step 4: segmentation ===")
    segment_tiles(cropped_root, segmented_root)

    print("\n=== Step 5: analysis ===")
    analyze_counts(segmented_root, analysis_root)

    print("\nPipeline complete. Results in:", out_root)

if __name__=="__main__":
    main()