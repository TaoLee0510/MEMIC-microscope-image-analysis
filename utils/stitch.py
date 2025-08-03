# stitch.py

import os
import numpy as np
import tifffile

# Same order & grid as in MEMIC.py
ORDER = [
    "f02","f03","f04","f05","f06","f07","f08","f09","f10",
    "f19","f18","f17","f16","f15","f14","f13","f12","f11",
    "f20","f21","f22","f23","f01","f24","f25","f26","f27",
    "f36","f35","f34","f33","f32","f31","f30","f29","f28",
    "f37","f38","f39","f40","f41","f42","f43","f44","f45"
]
ROWS, COLS = 5, 9

def stitch_images(edof_root: str, stitched_root: str):
    """
    Reads per-domain EDoF files from <edof_root>/<well_id>/<tp>/ and
    writes stitched BF, FL, and combined TIFFs into
    <stitched_root>/<well_id>/<tp>/.
    """
    for well_id in sorted(os.listdir(edof_root)):
        well_dir = os.path.join(edof_root, well_id)
        if not os.path.isdir(well_dir): continue
        for tp in sorted(os.listdir(well_dir)):
            tp_dir = os.path.join(well_dir, tp)
            if not os.path.isdir(tp_dir): continue

            dst_dir = os.path.join(stitched_root, well_id, tp)
            os.makedirs(dst_dir, exist_ok=True)

            # load domain images
            bf, fl = {}, {}
            for fname in os.listdir(tp_dir):
                if not fname.lower().endswith("_edof.tiff"):
                    continue
                domain, chan, _ = fname.split("_", 2)
                img = tifffile.imread(os.path.join(tp_dir, fname))
                (bf if chan=="bf" else fl)[domain] = img

            # pick sample tile to get dims & dtype
            sample = None
            for d in ORDER:
                if d in bf: sample = bf[d]; break
            if sample is None:
                for d in ORDER:
                    if d in fl: sample = fl[d]; break
            if sample is None:
                print(f"[stitch] skipping {well_id}/{tp}: no domain tiles")
                continue

            h, w = sample.shape
            ovx, ovy = int(w*0.05), int(h*0.05)
            W = w + (COLS-1)*(w-ovx)
            H = h + (ROWS-1)*(h-ovy)

            canvas_bf = np.zeros((H, W), dtype=sample.dtype)
            canvas_fl = np.zeros((H, W), dtype=sample.dtype)

            for r in range(ROWS):
                for c in range(COLS):
                    idx = r*COLS + c
                    key = ORDER[idx]
                    y0, x0 = r*(h-ovy), c*(w-ovx)
                    canvas_bf[y0:y0+h, x0:x0+w] = bf.get(key, np.zeros_like(sample))
                    canvas_fl[y0:y0+h, x0:x0+w] = fl.get(key, np.zeros_like(sample))

            # save
            bf_path = os.path.join(dst_dir, "stitched_image.tiff")
            tifffile.imwrite(bf_path, canvas_bf)
            print(f"[stitch] BF → {bf_path}")

            fl_path = os.path.join(dst_dir, "stitched_image_fl.tiff")
            tifffile.imwrite(fl_path, canvas_fl)
            print(f"[stitch] FL → {fl_path}")

            cmb_path = os.path.join(dst_dir, "stitched_combined.tiff")
            tifffile.imwrite(cmb_path, [canvas_bf, canvas_fl])
            print(f"[stitch] Combined → {cmb_path}")