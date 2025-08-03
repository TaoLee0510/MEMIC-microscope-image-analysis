
# segment.py

model = None  # global model to be initialized in each subprocess

import os
import cv2
import numpy as np
import tifffile
import random
from cellpose import models
from skimage.measure import regionprops
import gc
import concurrent.futures

from skimage.measure import label
from skimage.color import label2rgb

import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)


# Worker initializer for parallel tile processing
def _init_worker(model_path, use_gpu):
    global model
    model = models.CellposeModel(gpu=use_gpu, model_type=model_path)

# Helper for parallel tile processing
def _process_tile(args):
    global model
    tb, tf, x0, y0, sub_w, sub_h, diameter = args
    mb, *_ = model.eval(tb, diameter=diameter, channels=[0,0], flow_threshold=0.4, cellprob_threshold=0)
    mf, *_ = model.eval(tf, diameter=diameter, channels=[0,0], flow_threshold=0.4, cellprob_threshold=0)
    coords_b = [(int(p.centroid[1]+x0), int(p.centroid[0]+y0)) for p in regionprops(mb.astype(int))]
    coords_f = [(int(p.centroid[1]+x0), int(p.centroid[0]+y0)) for p in regionprops(mf.astype(int))]
    mask_b = (mb > 0)
    mask_f = (mf > 0)
    return coords_b, coords_f, x0, y0, sub_h, sub_w, mask_b, mask_f

def segment_tiles(cropped_root: str, segmented_root: str, diameter=30):
    """
    Splits each cropped BF/FL pair into random 10–14×10–14 overlapping tiles,
    runs Cellpose on each for BF and FL, stitches overlays & collects centroids.
    Saves in <segmented_root>/<well_id>/<tp>/:
      overlay_masks.tiff, cell_coordinates.csv, cell_coordinates_bf.csv, cell_coordinates_fl.csv, cellpose_evaluation.txt
    """
    #model = models.CellposeModel(gpu=False, model_type='cyto3')


    for well_id in sorted(os.listdir(cropped_root)):
        wd = os.path.join(cropped_root, well_id)
        if not os.path.isdir(wd): continue
        for tp in sorted(os.listdir(wd)):
            cd = os.path.join(wd, tp)
            if not os.path.isdir(cd): continue

            out = os.path.join(segmented_root, well_id, tp)
            os.makedirs(out, exist_ok=True)

            bf = tifffile.imread(os.path.join(cd, "cropped_image.tiff"))
            fl = tifffile.imread(os.path.join(cd, "cropped_image_fl.tiff"))
            H,W = bf.shape

            # Initialize separate masks for BF and FL for the entire image
            mosaic_bf = np.zeros((H, W), dtype=bool)
            mosaic_fl = np.zeros((H, W), dtype=bool)

            # random grid
            rows = random.randint(10,14)
            cols = random.randint(10,14)
            sub_w = int(W / (1+0.95*(cols-1)))
            sub_h = int(H / (1+0.95*(rows-1)))
            step_x = int(sub_w*0.95)
            step_y = int(sub_h*0.95)

            coords_bf = []
            coords_fl = []

            # Build a list of tile arguments
            tasks = []
            for i in range(rows):
                for j in range(cols):
                    x0 = min(j*step_x, W-sub_w)
                    y0 = min(i*step_y, H-sub_h)
                    tb = bf[y0:y0+sub_h, x0:x0+sub_w]
                    tf = fl[y0:y0+sub_h, x0:x0+sub_w]
                    tasks.append((tb, tf, x0, y0, sub_w, sub_h, diameter, None))

            # Reserve 1 CPU core for system, use remaining cores for segmentation
            num_cpus = max(1, os.cpu_count() - 2)
            # Process tiles in parallel
            #model_path = '/Volumes/Work_Active_1/MEMIC/models/cellpose_residual_default_on_style_on_concatenation_off_train_2021_08_24_17'
            model_path='cyto3'
            use_gpu = True
            tasks_args = [(tb, tf, x0, y0, sub_w, sub_h, diameter) for (tb, tf, x0, y0, sub_w, sub_h, diameter, _) in tasks]

            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=num_cpus,
                    initializer=_init_worker,
                    initargs=(model_path, use_gpu)) as executor:
                for coords_b_tile, coords_f_tile, x0, y0, sh, sw, mb_tile, mf_tile in executor.map(_process_tile, tasks_args):
                    coords_bf.extend(coords_b_tile)
                    coords_fl.extend(coords_f_tile)
                    mosaic_bf[y0:y0+sh, x0:x0+sw] |= mb_tile
                    mosaic_fl[y0:y0+sh, x0:x0+sw] |= mf_tile

            # coords CSVs
            def save_coords(lst, name):
                import csv
                with open(os.path.join(out,name), 'w', newline='') as f:
                    wtr=csv.writer(f); wtr.writerow(["X","Y"]); wtr.writerows(lst)
            save_coords(coords_bf,   "cell_coordinates_bf.csv")
            save_coords(coords_fl,   "cell_coordinates_fl.csv")

            # Save BF and FL binary masks
            bf_mask_path = os.path.join(out, "mask_bf.tiff")
            tifffile.imwrite(bf_mask_path, mosaic_bf.astype(np.uint8) * 255)
            print(f"[segment] saved BF mask → {bf_mask_path}")

            fl_mask_path = os.path.join(out, "mask_fl.tiff")
            tifffile.imwrite(fl_mask_path, mosaic_fl.astype(np.uint8) * 255)
            print(f"[segment] saved FL mask → {fl_mask_path}")

            # Create colored mask for BF and overlay on BF image
            labels_bf = label(mosaic_bf)
            # Generate color mask (float in [0,1])
            color_bf = label2rgb(labels_bf, image=None, bg_label=0)
            color_bf = (color_bf * 255).astype(np.uint8)  # to uint8 RGB
            # Convert BF image to uint8 and then to BGR
            bf_uint8 = cv2.convertScaleAbs(bf)
            bf_bgr = cv2.cvtColor(bf_uint8, cv2.COLOR_GRAY2BGR)
            # Blend original and mask
            overlay_bf = cv2.addWeighted(bf_bgr, 0.7, color_bf, 0.3, 0)
            overlay_bf_path = os.path.join(out, "overlay_bf.tiff")
            tifffile.imwrite(overlay_bf_path, overlay_bf)
            print(f"[segment] saved BF overlay → {overlay_bf_path}")

            # Create colored mask for FL and overlay on FL image
            labels_fl = label(mosaic_fl)
            color_fl = label2rgb(labels_fl, image=None, bg_label=0)
            color_fl = (color_fl * 255).astype(np.uint8)
            # Convert FL image to uint8 and then to BGR
            fl_uint8 = cv2.convertScaleAbs(fl)
            fl_bgr = cv2.cvtColor(fl_uint8, cv2.COLOR_GRAY2BGR)
            overlay_fl = cv2.addWeighted(fl_bgr, 0.7, color_fl, 0.3, 0)
            overlay_fl_path = os.path.join(out, "overlay_fl.tiff")
            tifffile.imwrite(overlay_fl_path, overlay_fl)
            print(f"[segment] saved FL overlay → {overlay_fl_path}")

            # Write summary statistics
            summary_path = os.path.join(out, "segmentation_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(f"BF cell count: {len(coords_bf)}\n")
                f.write(f"FL cell count: {len(coords_fl)}\n")
            print(f"[segment] saved summary → {summary_path}")

            print(f"[segment] done {well_id}/{tp}")
