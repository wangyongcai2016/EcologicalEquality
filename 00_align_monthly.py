#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 00 — Align monthly NPP / precipitation / PET onto the analysis grid (UNITS_TIF)
================================================================================
Why: the three monthly products sit on different grids (different origin, extent and
Albers latitude-of-origin: NPP 9840x9667, precipitation 9427x13908, PET 10175x14271,
analysis grid 8091x9668). 02_esi / 03_spei need pixel-for-pixel agreement, so each month
is resampled ONCE onto the UNITS_TIF grid (bilinear; GDAL ignores NaN neighbours) and
pixels outside the assessment units are set to NaN.

Run once before 02_esi.py (resumable: files that already exist are skipped).

INPUT  (geo_io.CONFIG): NPP_DIR / PRECIP_DIR / PET_DIR (+ *_NAME, *_PATTERN), UNITS_TIF,
                        START_YEAR / END_YEAR
OUTPUT : NPP_ALIGN_DIR / PRECIP_ALIGN_DIR / PET_ALIGN_DIR (same file names as the raw data;
         float32, NaN nodata, 256x256 tiles, deflate). Always the full grid (ROI is ignored here).
================================================================================
"""
from __future__ import annotations
import os, sys, time, warnings
import numpy as np
import geo_io as gio
from geo_io import CONFIG

warnings.filterwarnings("ignore")

_MASK = None


def _get_mask():
    global _MASK
    if _MASK is None:
        _MASK = gio.read_units(full=True) > 0
    return _MASK


def align_one(key, year, month):
    """Resample one month onto the analysis grid. Returns (status, path)."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    src_p = gio.monthly_path(key, year, month, aligned=False)
    dst_p = gio.monthly_path(key, year, month, aligned=True)
    if os.path.exists(dst_p):
        return "skip", dst_p
    if not os.path.exists(src_p):
        return "missing", src_p
    grid = gio.analysis_grid(full=True)
    dst = np.full((grid["height"], grid["width"]), np.nan, "float32")
    try:
        with rasterio.open(src_p) as s:
            reproject(rasterio.band(s, 1), dst,
                      src_transform=s.transform, src_crs=s.crs, src_nodata=s.nodata,
                      dst_transform=grid["transform"], dst_crs=grid["crs"], dst_nodata=np.nan,
                      resampling=Resampling.bilinear)
    except Exception as e:       # truncated / corrupt download: treat as a missing month
        return "unreadable", f"{src_p}  ({str(e)[:80]})"
    dst[~_get_mask()] = np.nan
    os.makedirs(os.path.dirname(dst_p), exist_ok=True)
    tmp = dst_p + ".part"
    prof = dict(driver="GTiff", dtype="float32", count=1, nodata=np.nan,
                height=grid["height"], width=grid["width"], crs=grid["crs"],
                transform=grid["transform"], tiled=True, blockxsize=256, blockysize=256,
                compress="deflate", predictor=3)
    with rasterio.open(tmp, "w", **prof) as d:
        d.write(dst, 1)
    os.replace(tmp, dst_p)
    return "done", dst_p


def main(keys=("npp", "pre", "pet")):
    from joblib import Parallel, delayed
    s, e = CONFIG["START_YEAR"], CONFIG["END_YEAR"]
    g = gio.analysis_grid(full=True)
    print(f"[ALIGN] target grid {g['height']}x{g['width']} (UNITS_TIF, full extent), {s}-{e}")
    tasks = [(k, y, m) for k in keys for y in range(s, e + 1) for m in range(1, 13)]
    t0 = time.time(); cnt = {"done": 0, "skip": 0, "missing": 0, "unreadable": 0}; missing = []
    gen = Parallel(n_jobs=CONFIG["N_JOBS_ALIGN"], return_as="generator_unordered")(
        delayed(align_one)(*t) for t in tasks)
    for i, (st, p) in enumerate(gen, 1):
        cnt[st] += 1
        if st in ("missing", "unreadable"):
            missing.append(p)
            print(f"[ALIGN] {st.upper()}: {p}", flush=True)
        if i % 100 == 0 or i == len(tasks):
            print(f"[ALIGN] {i}/{len(tasks)}  done {cnt['done']}  skipped {cnt['skip']}  "
                  f"missing {cnt['missing']}  unreadable {cnt['unreadable']}  {time.time() - t0:.0f}s", flush=True)
    per_key = {k: sum(1 for p in missing if os.path.basename(p.split("  (")[0]).lower().startswith(
        {"npp": "glass_npp", "pre": "pre", "pet": "pet"}[k])) for k in keys}
    over = {k: n for k, n in per_key.items() if n > CONFIG["MAX_MISSING_MONTHS"]}
    if over:
        raise FileNotFoundError(f"[ALIGN] too many missing months {over} "
                                f"(MAX_MISSING_MONTHS={CONFIG['MAX_MISSING_MONTHS']}); check dir/pattern in geo_io.py")
    if missing:
        print(f"[ALIGN] {len(missing)} missing month(s) will be filled by temporal interpolation in 02_esi.")
    print("[ALIGN] done.")


if __name__ == "__main__":
    main()
