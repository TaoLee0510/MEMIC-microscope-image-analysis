# analysis.py

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import tifffile

def analyze_counts(segmented_root: str, analysis_root: str, bin_size=500):
    """
    For each overlay under <segmented_root>/<well_id>/<tp>/, loads
    cell_coordinates*.csv and produces:
      - cell_counts_bf.csv, cell_count_bar_bf.png
      - cell_counts_fl.csv, cell_count_bar_fl.png
      - cell_counts.csv,    cell_count_bar.png
    into <analysis_root>/<well_id>/<tp>/.
    """
    for well_id in sorted(os.listdir(segmented_root)):
        wd = os.path.join(segmented_root, well_id)
        if not os.path.isdir(wd): continue
        for tp in sorted(os.listdir(wd)):
            sd = os.path.join(wd, tp)
            if not os.path.isdir(sd): continue

            out = os.path.join(analysis_root, well_id, tp)
            os.makedirs(out, exist_ok=True)

            # load coords
            def load(n): 
                path = os.path.join(sd, n)
                if os.path.exists(path):
                    arr = np.loadtxt(path, delimiter=',', skiprows=1)
                    return arr[:,0] if arr.ndim>1 else np.array([])
                return np.array([])
            x_bf = load("cell_coordinates_bf.csv")
            x_fl = load("cell_coordinates_fl.csv")
            x_cb = load("cell_coordinates.csv")

            # common bins
            max_x = max(x_cb.max() if x_cb.size else 0,
                        x_bf.max() if x_bf.size else 0,
                        x_fl.max() if x_fl.size else 0)
            bins = np.arange(0, max_x+bin_size, bin_size)
            centers = (bins[:-1]+bins[1:])/2

            for name,x in [("bf", x_bf), ("fl", x_fl), ("combined", x_cb)]:
                counts,_ = np.histogram(x, bins)
                # CSV
                csvp = os.path.join(out, f"cell_counts_{name}.csv")
                with open(csvp,'w', newline='') as f:
                    wtr=csv.writer(f)
                    wtr.writerow(["Bin_Start","Bin_End","Count"])
                    for i in range(len(counts)):
                        wtr.writerow([bins[i], bins[i+1], counts[i]])
                # bar
                plt.figure(figsize=(8,5))
                plt.bar(centers, counts, width=bin_size, align='center')
                plt.title(f"{name.upper()} Cell Count vs X-position")
                plt.xlabel("X (px)"); plt.ylabel("Count")
                pngp = os.path.join(out, f"cell_count_bar_{name}.png")
                plt.savefig(pngp)
                plt.close()
                print(f"[analyze] {well_id}/{tp}/{name} → {csvp}, {pngp}")