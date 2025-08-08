#!/usr/bin/env python3
# deps: numpy, tifffile, scikit-image, torch, cellpose>=4
import os, re, argparse, random, traceback, threading
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tifffile import imread, imwrite, TiffFile
from skimage.filters import sobel, laplace, gaussian
from skimage.segmentation import find_boundaries
from skimage.transform import pyramid_laplacian, resize

# -------------------- filename parsing --------------------
PAT = re.compile(r"r(?P<r>\d+)c(?P<c>\d+)f(?P<f>\d+)p(?P<p>\d+)-ch(?P<ch>\d+)sk(?P<sk>\d+)fk\d+fl\d+\.tif{1,2}f$", re.IGNORECASE)
TAG_RE = re.compile(r"(r\d{2}c\d{2}f\d{2}sk\d+)", re.IGNORECASE)

METHODS = [
    "BESTSLICE",   # Tenengrad best single slice
    "MIP",         # Max intensity projection
    "EDF_HARD",    # Laplacian argmax
    "EDF_GAUSS",   # Laplacian -> Gaussian -> argmax
    "EDF_IJ",      # ImageJ-like multiscale fusion (Laplacian pyramid)
    "PLANE_FIT"    # Fit plane to coarse z-map then sample
]

# -------------------- discovery --------------------
def parse_fname(fname):
    m = PAT.fullmatch(os.path.basename(fname))
    if not m: return None
    g = {k:int(v) for k,v in m.groupdict().items()}
    g["fname"] = fname
    return g

def list_groups(indir):
    files = [os.path.join(indir, f) for f in os.listdir(indir) if f.lower().endswith((".tif",".tiff"))]
    meta = [parse_fname(f) for f in files]
    meta = [m for m in meta if m]
    groups = {}
    for m in meta:
        key = (m["r"], m["c"], m["f"], m["sk"])
        groups.setdefault(key, []).append(m)
    return groups

# -------------------- IO helpers --------------------
def im_to_u16(a):
    return a if a.dtype == np.uint16 else np.rint(np.clip(a, 0, 65535)).astype(np.uint16)

def save_imagej_yxc(path, yxc, labels=None, verbose=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cyx = np.moveaxis(yxc, -1, 0)[np.newaxis, np.newaxis, ...]  # 1,1,C,Y,X
    imwrite(path, im_to_u16(cyx), imagej=True, photometric="minisblack",
            metadata={"Labels": labels or []})
    if verbose:
        with TiffFile(path) as tf:
            s = tf.series[0]
            print(f"[WRITE] {os.path.basename(path)} axes={s.axes} shape={s.shape}", flush=True)

def load_stack_for(items, ch):
    sel = sorted([m for m in items if m["ch"]==ch], key=lambda x:x["p"])
    if not sel: return None
    arr = np.stack([imread(m["fname"]) for m in sel], axis=0)  # Z,H,W
    return arr, [m["p"] for m in sel]

def norm_yxc(arr):
    if arr.ndim==5 and arr.shape[:2]==(1,1): return np.moveaxis(arr[0,0],0,-1)
    if arr.ndim==3 and arr.shape[0] in (1,2,3,4) and arr.shape[-1] not in (1,2,3,4): return np.moveaxis(arr,0,-1)
    if arr.ndim==3 and arr.shape[-1] in (1,2,3,4): return arr
    raise ValueError(f"Unexpected shape {arr.shape}")

# -------------------- focus/util --------------------
def tenengrad_score(img):
    g = sobel(img.astype(np.float32)); return float(np.mean(g*g))

def argmax_gather(zmap, stacks):
    H,W = zmap.shape
    rr = np.arange(H)[:,None]; cc = np.arange(W)[None,:]
    return np.stack([s[(zmap, rr, cc)] for s in stacks], axis=-1)

# -------------------- methods --------------------
# 1) Best slice (global Tenengrad on ch1)
def m_bestslice(stack_ref, stacks):
    scores = [tenengrad_score(stack_ref[z]) for z in range(stack_ref.shape[0])]
    z = int(np.argmax(scores))
    return np.stack([s[z] for s in stacks], axis=-1), z

# 2) Max intensity projection
def m_mip(stacks):
    return np.stack([np.max(s, axis=0) for s in stacks], axis=-1)

# 3) EDF_HARD: per-pixel argmax of |Laplacian|
def m_edf_hard(stack_ref, stacks):
    L = np.abs(laplace(stack_ref.astype(np.float32)))
    zmap = np.argmax(L, axis=0)
    return argmax_gather(zmap, stacks)

# 4) EDF_GAUSS: Laplacian -> Gaussian (spatial) -> argmax
def m_edf_gauss(stack_ref, stacks, sigma=0.8, truncate=3.0):
    L = np.abs(laplace(stack_ref.astype(np.float32)))
    Ls = np.stack([gaussian(L[z], sigma=sigma, truncate=truncate, preserve_range=True) for z in range(L.shape[0])], axis=0)
    zmap = np.argmax(Ls, axis=0)
    return argmax_gather(zmap, stacks)

# 5) EDF_IJ: multiscale Laplacian-pyramid fusion (fixed)
def build_lap_pyr(img, max_levels):
    return list(pyramid_laplacian(img, max_layer=max_levels-1, downscale=2, mode='reflect'))

def reconstruct_from_lap_pyr(pyr):
    rec = pyr[-1]
    for lev in range(len(pyr)-2, -1, -1):
        up = resize(rec, pyr[lev].shape, order=1, mode="reflect",
                    anti_aliasing=False, preserve_range=True)
        rec = up + pyr[lev]
    return rec

def m_edf_ij(stack_ref, stacks, levels=4):
    """ImageJ-like multiscale fusion: per-level winner-take-all on ref, apply to both channels."""
    Z, H, W = stack_ref.shape
    levels = max(2, min(levels, int(np.floor(np.log2(min(H, W))))))

    # pyramids for ref channel
    ref_pyrs = [build_lap_pyr(stack_ref[z].astype(np.float32), levels) for z in range(Z)]
    nlev = len(ref_pyrs[0])  # includes lowpass

    # selection maps per level (at that level's resolution)
    zsel = []
    for lev in range(nlev-1):  # detail levels
        A = np.stack([np.abs(ref_pyrs[z][lev]) for z in range(Z)], axis=0)  # (Z,h,w)
        zlev = np.argmax(A, axis=0).astype(np.int16)                         # (h,w)
        zsel.append(zlev)
    # lowpass level selection at lowpass resolution
    A_low = np.stack([np.abs(ref_pyrs[z][-1]) for z in range(Z)], axis=0)    # (Z,hL,wL)
    zlow = np.argmax(A_low, axis=0).astype(np.int16)                         # (hL,wL)

    outs = []
    for ch_stack in stacks:
        pyrs = [build_lap_pyr(ch_stack[z].astype(np.float32), levels) for z in range(Z)]
        fused = []
        # fuse detail levels
        for lev in range(nlev-1):
            B = np.stack([pyrs[z][lev] for z in range(Z)], axis=0)           # (Z,h,w)
            sel = zsel[lev][None, ...]                                       # (1,h,w)
            fused_level = np.take_along_axis(B, sel, axis=0)[0]              # (h,w)
            fused.append(fused_level)
        # fuse lowpass
        B_low = np.stack([pyrs[z][-1] for z in range(Z)], axis=0)            # (Z,hL,wL)
        fused_low = np.take_along_axis(B_low, zlow[None, ...], axis=0)[0]    # (hL,wL)
        fused.append(fused_low)
        rec = reconstruct_from_lap_pyr(fused)
        outs.append(rec)
    return np.stack(outs, axis=-1)

# 6) PLANE_FIT: fit z(x,y) = ax+by+c from coarse z-map (EDF_GAUSS) then sample
def m_plane_fit(stack_ref, stacks, sigma=0.8, grid=32):
    L = np.abs(laplace(stack_ref.astype(np.float32)))
    Ls = np.stack([gaussian(L[z], sigma=sigma, preserve_range=True) for z in range(L.shape[0])], axis=0)
    zmap = np.argmax(Ls, axis=0).astype(np.float32)
    H,W = zmap.shape
    ys = np.linspace(0, H-1, num=max(2, H//grid), dtype=np.float32)
    xs = np.linspace(0, W-1, num=max(2, W//grid), dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    zz = zmap[yy.astype(int), xx.astype(int)]
    X = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size, dtype=np.float32)], axis=1)
    coef, *_ = np.linalg.lstsq(X, zz.ravel(), rcond=None)
    a, b, c = coef
    YY, XX = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing='ij')
    zsurf = a*XX + b*YY + c
    zsurf = np.rint(np.clip(zsurf, 0, stack_ref.shape[0]-1)).astype(np.int32)
    return argmax_gather(zsurf, stacks)

# -------------------- Phase 1 worker: flatten (parallel) --------------------
def flatten_worker(task, outroot, sigma_gauss, ij_levels, plane_grid):
    (r,c,f,sk), items = task
    tag = f"r{r:02d}c{c:02d}f{f:02d}sk{sk}"
    try:
        (s1,p1) = load_stack_for(items, 1); (s2,p2) = load_stack_for(items, 2)
        if s1 is None or s2 is None: return f"[SKIP] {tag} missing ch"
        if p1 != p2: return f"[SKIP] {tag} mismatched Z"
        s1 = s1.astype(np.float32); s2 = s2.astype(np.float32)
        stacks = [s1, s2]
        outdir = os.path.join(outroot, f"r{r:02d}c{c:02d}", f"f{f:02d}", f"sk{sk}")
        base = f"r{r:02d}c{c:02d}f{f:02d}sk{sk}"
        os.makedirs(outdir, exist_ok=True)

        img, z = m_bestslice(s1, stacks)
        save_imagej_yxc(os.path.join(outdir, f"{base}-BESTSLICE_p{p1[z]:02d}.tiff"), img, ["ch1","ch2"])

        img = m_mip(stacks)
        save_imagej_yxc(os.path.join(outdir, f"{base}-MIP.tiff"), img, ["ch1","ch2"])

        img = m_edf_hard(s1, stacks)
        save_imagej_yxc(os.path.join(outdir, f"{base}-EDF_HARD.tiff"), img, ["ch1","ch2"])

        img = m_edf_gauss(s1, stacks, sigma=sigma_gauss)
        save_imagej_yxc(os.path.join(outdir, f"{base}-EDF_GAUSS_s{sigma_gauss}.tiff"), img, ["ch1","ch2"])

        img = m_edf_ij(s1, stacks, levels=ij_levels)
        save_imagej_yxc(os.path.join(outdir, f"{base}-EDF_IJ_L{ij_levels}.tiff"), img, ["ch1","ch2"])

        img = m_plane_fit(s1, stacks, sigma=sigma_gauss, grid=plane_grid)
        save_imagej_yxc(os.path.join(outdir, f"{base}-PLANE_FIT.tiff"), img, ["ch1","ch2"])

        return f"[OK] {tag}"
    except Exception as e:
        return f"[ERR] {tag}: {e}\n{traceback.format_exc(limit=2)}"

# -------------------- Phase 2: CellposeSAM threaded (shared model) --------------------
_model = None
_model_lock = threading.Lock()
_eval_lock = threading.Lock()

def get_model(pretrained_model=None):
    global _model
    if _model is None:
        import torch
        from cellpose.models import CellposeModel
        with _model_lock:
            if _model is None:
                if pretrained_model:
                    _model = CellposeModel(gpu=torch.cuda.is_available(), pretrained_model=pretrained_model)
                else:
                    _model = CellposeModel(gpu=torch.cuda.is_available())
    return _model

def cpsam_boundaries(yxc, pretrained_model=None, lock_gpu=True):
    model = get_model(pretrained_model=pretrained_model)
    if lock_gpu:
        with _eval_lock:
            masks, _, _ = model.eval(yxc, diameter=None)
    else:
        masks, _, _ = model.eval(yxc, diameter=None)
    if masks is None or masks.size == 0:
        return np.zeros(yxc.shape[:2], dtype=np.uint16)
    b = find_boundaries(masks, mode='inner')
    return (b.astype(np.uint16) * 65535)

def upgrade_to_c3(path_2c, out_path, pretrained_model=None, lock_gpu=True):
    yxc = norm_yxc(imread(path_2c))          # Y,X,2
    b = cpsam_boundaries(yxc, pretrained_model=pretrained_model, lock_gpu=lock_gpu)[..., None]  # Y,X,1
    c3 = np.dstack([yxc, b])                 # Y,X,3
    save_imagej_yxc(out_path, c3, ["ch1","ch2","mask_bnd"])
    return out_path

# -------------------- tiling/panel --------------------
def find_method_files(indir, want_c3=True):
    groups = {}
    for root, _, files in os.walk(indir):
        for fn in files:
            if not fn.lower().endswith((".tif",".tiff")): continue
            if want_c3 and "_C3." not in fn.upper(): continue
            m = TAG_RE.search(fn)
            if not m: continue
            tag = m.group(1)
            up = fn.upper()
            for mk in METHODS:
                if mk in up:
                    groups.setdefault(tag, {})[mk] = os.path.join(root, fn)
    return groups

def tile_grid(imgs, cols, rows):
    Y,X,C = imgs[0].shape
    for im in imgs:
        if im.shape != (Y,X,C): raise SystemExit("size mismatch")
    out = []; k = 0
    for _ in range(rows):
        out.append(np.concatenate(imgs[k:k+cols], axis=1)); k += cols
    return np.concatenate(out, axis=0)

def save_panel(output_path, mosaics, labels):
    z_c_yx = np.stack([np.moveaxis(m, -1, 0) for m in mosaics], axis=0)  # Z,C,Y,X
    data = im_to_u16(z_c_yx)[np.newaxis, ...]  # 1,Z,C,Y,X
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    imwrite(output_path, data, imagej=True, photometric="minisblack", metadata={"Labels": labels})
    with TiffFile(output_path) as tf:
        s = tf.series[0]
        print(f"[PANEL] axes={s.axes} shape={s.shape}")

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outroot", required=True)
    ap.add_argument("--panel_out", required=True)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="process workers for flattening")
    ap.add_argument("--cpose_workers", type=int, default=1, help="thread workers for Cellpose phase")
    ap.add_argument("--allow_parallel_cpose", action="store_true", help="allow multiple GPU evals concurrently (risk OOM)")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=3)
    # Defaults tuned for your case (2k×2k, smooth ±2–3 z)
    ap.add_argument("--sigma_gauss", type=float, default=0.8, help="Gaussian sigma for EDF_GAUSS & plane pre-map")
    ap.add_argument("--ij_levels", type=int, default=4, help="pyramid levels for EDF_IJ")
    ap.add_argument("--plane_grid", type=int, default=32, help="grid for plane fit (larger=faster)")
    ap.add_argument("--finetuned_cellpose_model", default=None, help="optional path to finetuned Cellpose model")
    args = ap.parse_args()

    # sample
    random.seed(args.seed)
    groups = list_groups(args.input)
    if not groups: raise SystemExit("No files matched expected pattern.")
    keys = list(groups.keys()); random.shuffle(keys); keys = keys[:args.samples]
    print(f"[INFO] Flattening {len(keys)} groups (of {len(groups)}) with {args.workers} workers.", flush=True)

    # Phase 1: flatten in parallel (2-channel outputs)
    tasks = [((r,c,f,sk), groups[(r,c,f,sk)]) for (r,c,f,sk) in keys]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(flatten_worker, t, args.outroot, args.sigma_gauss, args.ij_levels, args.plane_grid) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            print(f"[FLAT {i}/{len(futs)}] {fut.result()}", flush=True)

    # Phase 2: CellposeSAM threaded with shared model (GPU serialized by default)
    print("[INFO] Discovering 2C outputs for Cellpose…", flush=True)
    two_c = []
    for root,_,files in os.walk(args.outroot):
        for fn in files:
            if not fn.lower().endswith((".tif",".tiff")): continue
            up = fn.upper()
            if any(m in up for m in METHODS) and "_C3." not in up:
                two_c.append(os.path.join(root, fn))
    two_c.sort()
    print(f"[INFO] CellposeSAM on {len(two_c)} images with {args.cpose_workers} thread(s).", flush=True)

    lock_gpu = not args.allow_parallel_cpose
    total = len(two_c)
    done = 0
    done_lock = threading.Lock()

    # warm up model once
    get_model(pretrained_model=args.finetuned_cellpose_model)

    def _job(path2c):
        nonlocal done
        outp = path2c.replace(".tiff","_C3.tiff").replace(".tif","_C3.tif")
        try:
            upgrade_to_c3(path2c, outp, pretrained_model=args.finetuned_cellpose_model, lock_gpu=lock_gpu)
        except Exception as e:
            outp = f"[ERR] {os.path.basename(path2c)}: {e}"
        with done_lock:
            done += 1
            print(f"[CP {done}/{total}] {os.path.basename(outp)}", flush=True)
        return outp

    with ThreadPoolExecutor(max_workers=args.cpose_workers) as tpex:
        list(tpex.map(_job, two_c))

    # Build panel from *_C3.* only (Z=6, C=3)
    need = args.cols * args.rows
    groups_map = {}
    for root,_,files in os.walk(args.outroot):
        for fn in files:
            if not fn.lower().endswith((".tif",".tiff")): continue
            if "_C3." not in fn.upper(): continue
            m = TAG_RE.search(fn)
            if not m: continue
            tag = m.group(1)
            up = fn.upper()
            for mk in METHODS:
                if mk in up:
                    groups_map.setdefault(tag, {})[mk] = os.path.join(root, fn)

    complete = [(tag, files) for tag,files in groups_map.items() if all(m in files for m in METHODS)]
    if len(complete) < need:
        raise SystemExit(f"Need {need} complete groups with C3 outputs, found {len(complete)}.")
    complete.sort(key=lambda x: x[0]); complete = complete[:need]

    per_method = {m: [] for m in METHODS}
    for tag, files in complete:
        for m in METHODS:
            per_method[m].append(norm_yxc(imread(files[m])))

    ref = per_method[METHODS[0]][0].shape
    for m, lst in per_method.items():
        for im in lst:
            if im.shape != ref: raise SystemExit(f"Size mismatch in {m}: {im.shape} vs {ref}")

    mosaics = [tile_grid(per_method[m], args.cols, args.rows) for m in METHODS]
    save_panel(args.panel_out, mosaics, labels=["ch1","ch2","mask_bnd"])
    print("[DONE]", flush=True)

if __name__ == "__main__":
    main()

