#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 00 — Human Footprint: zip archives -> annual 500 m Albers GeoTIFFs on the analysis grid
================================================================================
Source: annual global Human Footprint, 1 km, Mollweide (ESRI:54009), values 0-50 (Mu et al. 2022 for 2000-2018;
an operational update with a different file layout for 2019 on: LZW tiled, NaN nodata, vs. uncompressed, -inf nodata).
One zip per year in CONFIG["HFP_ZIP_DIR"] (hfp<year>.zip holding hfp<year>.tif).

For every year: extract to HFP_TMP, bilinear-resample onto the UNITS_TIF grid (500 m Albers; Mollweide is equal-area,
so the index needs no density correction; -inf/NaN/ocean -> NaN), write HFP_DIR/HFP_Mean_500m_Albers_<year>_clip.tif
(float32, nodata -9999, same layout as the FVC / LAI / GPP alignments), delete the extracted copy.

Also writes OUT_DIR/hfp_national_mean.csv: mean HFP of the analysed (stable natural) pixels per year, and the year-
to-year change, to expose a product break between the 2018 and 2019 vintages. The check is printed; it does not alter
the data.

Run:  python 00_prepare_hfp.py            (all years, skips files that exist)    EQI_HFP_YEARS=2019,2020 to limit.
================================================================================
"""
from __future__ import annotations
import os, sys, csv, time, zipfile, shutil, warnings
import numpy as np
import geo_io as gio
from geo_io import CONFIG

warnings.filterwarnings("ignore", category=RuntimeWarning)
NODATA = -9999.0


def out_path(year):
    return os.path.join(CONFIG["HFP_DIR"], CONFIG["HFP_PATTERN"].format(year=year))


def process_year(year):
    import rasterio
    from rasterio.warp import reproject, Resampling, transform_bounds
    from rasterio.windows import from_bounds
    dst = out_path(year)
    if os.path.exists(dst):
        return year, "exists", 0.0
    t0 = time.time()
    zp = os.path.join(CONFIG["HFP_ZIP_DIR"], f"hfp{year}.zip")
    if not os.path.exists(zp):
        return year, "no zip", 0.0
    tmp = os.path.join(CONFIG["HFP_TMP"], str(year)); os.makedirs(tmp, exist_ok=True)
    with zipfile.ZipFile(zp) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".tif")]
        z.extract(names[0], tmp)
    src_path = os.path.join(tmp, names[0])
    grid = gio.analysis_grid(full=True)
    H, W = grid["height"], grid["width"]
    with rasterio.open(src_path) as s:
        # source window covering the analysis grid (+ margin of 5 source pixels), read in memory
        l, b, r, t = rasterio.transform.array_bounds(H, W, grid["transform"])
        sb = transform_bounds(grid["crs"], s.crs, l, b, r, t, densify_pts=41)
        win = from_bounds(*sb, transform=s.transform).round_offsets().round_lengths()
        win = win.intersection(rasterio.windows.Window(0, 0, s.width, s.height))
        pad = 5
        win = rasterio.windows.Window(max(win.col_off - pad, 0), max(win.row_off - pad, 0),
                                      min(win.width + 2 * pad, s.width), min(win.height + 2 * pad, s.height))
        a = s.read(1, window=win).astype("float32")
        a[~np.isfinite(a)] = np.nan
        st = s.window_transform(win); scrs = s.crs
    out = np.full((H, W), np.nan, "float32")
    reproject(a, out, src_transform=st, src_crs=scrs, src_nodata=np.nan,
              dst_transform=grid["transform"], dst_crs=grid["crs"], dst_nodata=np.nan,
              resampling=Resampling.bilinear)
    os.makedirs(CONFIG["HFP_DIR"], exist_ok=True)
    prof = grid.copy(); prof.update(count=1, dtype="float32", nodata=NODATA, compress="lzw", tiled=True,
                                    blockxsize=256, blockysize=256)
    prof.pop("photometric", None)
    w = np.where(np.isfinite(out), out, NODATA).astype("float32")
    with rasterio.open(dst + ".part", "w", **prof) as d:
        d.write(w, 1)
        d.update_tags(source=f"hfp{year}.zip", resampling="bilinear", range="0-50")
    os.replace(dst + ".part", dst)
    shutil.rmtree(tmp, ignore_errors=True)
    return year, "ok", time.time() - t0


def national_means(years):
    import rasterio
    mask = gio.read_mask()
    rows = []
    for y in years:
        p = out_path(y)
        if not os.path.exists(p):
            continue
        with rasterio.open(p) as s:
            a = s.read(1).astype("float32")
        a[a == NODATA] = np.nan
        v = a[mask & np.isfinite(a)]
        rows.append([y, int(v.size), float(v.mean()), float(np.median(v)), float((a[mask] == 0).mean() * 100)])
    return rows


def main():
    years = list(range(CONFIG["START_YEAR"], CONFIG["END_YEAR"] + 1))
    if os.environ.get("EQI_HFP_YEARS"):
        years = [int(v) for v in os.environ["EQI_HFP_YEARS"].split(",")]
    todo = [y for y in years if not os.path.exists(out_path(y))]
    print(f"[HFP] {len(years)} years requested, {len(todo)} to process -> {CONFIG['HFP_DIR']}", flush=True)
    os.makedirs(CONFIG["HFP_TMP"], exist_ok=True)
    nj = int(os.environ.get("EQI_HFP_JOBS", 4))
    if todo:
        from joblib import Parallel, delayed
        t0 = time.time()
        gen = Parallel(n_jobs=min(nj, len(todo)), return_as="generator_unordered")(delayed(process_year)(y) for y in todo)
        for y, st, dt in gen:
            print(f"[HFP]   {y}: {st} ({dt:.0f}s)  elapsed {time.time() - t0:.0f}s", flush=True)
    shutil.rmtree(CONFIG["HFP_TMP"], ignore_errors=True)

    rows = national_means(years)
    out = gio.ensure_outdir()
    with open(os.path.join(out, "hfp_national_mean.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["year", "n_px", "mean_hfp", "median_hfp", "zero_pct", "diff_vs_prev"])
        prev = None
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.4f}", f"{r[4]:.2f}", "" if prev is None else f"{r[2] - prev:+.4f}"])
            prev = r[2]
    print("[HFP] mean HFP over the analysed pixels:")
    prev = None
    for r in rows:
        print(f"        {r[0]}  mean {r[2]:7.4f}  median {r[3]:7.4f}  {'' if prev is None else f'd={r[2] - prev:+.4f}'}")
        prev = r[2]
    d = np.diff([r[2] for r in rows])
    ys = [r[0] for r in rows][1:]
    if len(d) > 6:
        base = np.median(np.abs(d)); big = [(y, x) for y, x in zip(ys, d) if abs(x) > 4 * max(base, 1e-9)]
        print(f"[HFP] typical |year-to-year change| {base:.4f}; years with |change| > 4x that: "
              f"{[(y, round(float(x), 4)) for y, x in big] or 'none'}")
    print("[HFP] done.")


if __name__ == "__main__":
    main()
