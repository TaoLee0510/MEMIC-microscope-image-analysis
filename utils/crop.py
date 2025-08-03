# crop.py

import os
import cv2
import numpy as np
import tifffile

def crop_images(stitched_root: str, cropped_root: str):
    """
    For each stitched image under <stitched_root>/<well_id>/<tp>/,
    compute local-variance border and crop both BF and FL, saving:
      cropped_image.tiff, cropped_image_fl.tiff, cropped_combined.tiff
    into <cropped_root>/<well_id>/<tp>/.
    """
    for well_id in sorted(os.listdir(stitched_root)):
        wdir = os.path.join(stitched_root, well_id)
        if not os.path.isdir(wdir): continue
        for tp in sorted(os.listdir(wdir)):
            sdir = os.path.join(wdir, tp)
            if not os.path.isdir(sdir): continue

            dst = os.path.join(cropped_root, well_id, tp)
            os.makedirs(dst, exist_ok=True)

            bf = tifffile.imread(os.path.join(sdir, "stitched_image.tiff"))
            fl = tifffile.imread(os.path.join(sdir, "stitched_image_fl.tiff"))
            gray = bf.astype(np.float32)

            # local variance
            k = 31
            m = cv2.blur(gray, (k, k))
            m2 = cv2.blur(gray*gray, (k, k))
            var = m2 - m*m
            uv = cv2.normalize(var, None, 0,255, cv2.NORM_MINMAX).astype(np.uint8)
            _, mask = cv2.threshold(uv, int(uv.mean()), 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)),
                                    iterations=2)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                x,y,w0,h0 = cv2.boundingRect(max(cnts, key=cv2.contourArea))
                pad = 10
                x,y = max(x-pad,0), max(y-pad,0)
                w0 = min(w0+2*pad, bf.shape[1]-x)
                h0 = min(h0+2*pad, bf.shape[0]-y)
                cbf = bf[y:y+h0, x:x+w0]
                cfl = fl[y:y+h0, x:x+w0]
            else:
                cbf, cfl = bf.copy(), fl.copy()

            tifffile.imwrite(os.path.join(dst, "cropped_image.tiff"), cbf)
            tifffile.imwrite(os.path.join(dst, "cropped_image_fl.tiff"), cfl)
            tifffile.imwrite(os.path.join(dst, "cropped_combined.tiff"), [cbf, cfl])
            print(f"[crop] {well_id}/{tp} → cropped")