#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 09a — Build the attribution tables for 09_attribution.py from the pipeline outputs and driver rasters
================================================================================
Random sample of analysed pixels with a complete EFI / ESI / EPI / EQI series in the core years (05_eqi: normally
2002-2020). For every sampled pixel the CHANGE of each variable is its Theil-Sen slope per year over the core years:
    EFI_change ESI_change EPI_change EQI_change   slopes of efi_stack / esi_stack (window centres) / epi_stack / eqi_stack
    AP_change  PET_change AI_change               annual precipitation / PET / aridity (E:\\...\\Pre_Year, PET_Year, ai_YYYY.tif,
                                                  1 km WGS84, read at the pixel centre)
    AT_change                                     annual mean of the 12 monthly 500 m temperature files (read on a 1/4
                                                  decimated grid, ~2 km: temperature is smooth), CONFIG CLIM_DIRS["tmp"]
    FP_change                                     Human Footprint slope over HFP_YEARS (2002-2018 by default): the 2019
                                                  vintage is ~0.33 higher nationally (00_prepare_hfp.py), so the series is NOT
                                                  continued past 2018; the slope is per year over that shorter window
    Lu_change                                     change of the human-use share (ESA-CCI cropland 10/11/12/20, mosaic 30/40,
                                                  urban 190) in a ~3 km block, 2015 minus 2000. Both years come from the
                                                  v2.0.7 production line (the 2015->2016 product break is avoided).
Outputs (OUT_DIR): attribution_table.csv (N_SEM rows), mgwr_table.csv (N_MGWR rows: x, y in the analysis CRS, dEQI,
precip, temp, landuse, footprint = the changes above), attribution_sample_info.txt.
Run:  python 09a_sample_attribution.py        env EQI_ATTR_N (default 40000), EQI_ATTR_NMGWR (default 2000)
================================================================================
"""
from __future__ import annotations
import os, glob, time, importlib.util, warnings
import numpy as np
import geo_io as gio
from geo_io import CONFIG

warnings.filterwarnings("ignore", category=RuntimeWarning)
PER_TYPE = int(os.environ.get("EQI_ATTR_PERTYPE", 10000))
N_SEM = int(os.environ.get("EQI_ATTR_N", 40000))
N_MGWR = int(os.environ.get("EQI_ATTR_NMGWR", 2000))
SEED = 909
HFP_YEARS = (2002, 2018)
LU_YEARS = (2000, 2015)
HUMAN_CODES = [10, 11, 12, 20, 30, 40, 190]
HERE = os.path.dirname(os.path.abspath(__file__))
CLIM = r"E:\中国1km分辨率逐月气温和降水"


def ts(Y, x=None):
    """Theil-Sen slope per column of Y (T, n); x = time (default 0..T-1); NaN if fewer than 5 finite values."""
    T = Y.shape[0]; x = np.arange(T, dtype="float64") if x is None else np.asarray(x, float)
    ii, jj = np.triu_indices(T, 1)
    S = (Y[jj] - Y[ii]) / (x[jj] - x[ii])[:, None]
    sl = np.nanmedian(S, axis=0)
    sl[np.isfinite(Y).sum(0) < 5] = np.nan
    return sl


def at_pts(arr, idx, rows, cols):
    out = np.empty((len(idx), rows.size), "float32")
    for k, t in enumerate(idx):
        out[k] = np.asarray(arr[t][rows, cols])
    return out


def lonlat(grid, rows, cols):
    from pyproj import Transformer
    x = grid["transform"].c + (cols + 0.5) * grid["transform"].a
    y = grid["transform"].f + (rows + 0.5) * grid["transform"].e
    tr = Transformer.from_crs(grid["crs"], "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(x, y)
    return np.asarray(lon), np.asarray(lat), x, y


def read_at_lonlat(path, lon, lat, decim=1):
    """Value at (lon, lat) from a geographic raster; decim > 1 reads a nearest-decimated grid. NaN for nodata."""
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.open(path) as s:
        if decim > 1:
            a = s.read(1, out_shape=(s.height // decim, s.width // decim), resampling=Resampling.nearest)
            tr = s.transform * s.transform.scale(s.width / a.shape[1], s.height / a.shape[0])
        else:
            a = s.read(1); tr = s.transform
        nod = s.nodata; crs = s.crs
    a = a.astype("float64")
    if nod is not None and np.isfinite(nod):
        a[a == nod] = np.nan
    a[~np.isfinite(a) | (np.abs(a) > 1e30)] = np.nan
    return a, tr, crs


def sample_geo(path, lon, lat, decim=1):
    import rasterio.transform as rt
    from pyproj import Transformer
    a, tr, crs = read_at_lonlat(path, lon, lat, decim)
    if crs.to_string() != "EPSG:4326":
        x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    else:
        x, y = lon, lat
    r, c = rt.rowcol(tr, x, y)
    r, c = np.asarray(r), np.asarray(c)
    ok = (r >= 0) & (r < a.shape[0]) & (c >= 0) & (c < a.shape[1])
    out = np.full(lon.size, np.nan)
    out[ok] = a[r[ok], c[ok]]
    return out


def temp_year(args):
    year, lon, lat = args
    fs = sorted(glob.glob(os.path.join(CONFIG["CLIM_DIRS"]["tmp"], f"tmp_{year}_*_500m.tif")))
    fs = [f for f in fs if f.endswith(".tif")]
    if len(fs) != 12:
        return year, np.full(lon.size, np.nan)
    m = np.stack([sample_geo(f, lon, lat, decim=4) for f in fs])
    return year, m.mean(0)


def human_share(year, lon, lat, block=10):
    """Share of ESA-CCI human-use 300 m pixels in block x block cells (~3 km), read at (lon, lat)."""
    import rasterio
    from rasterio.windows import Window
    f = glob.glob(os.path.join(CONFIG["LANDUSE_RAW_DIR"], f"*-{year}-*.tif"))[0]
    with rasterio.open(f) as s:
        res = s.res[0]
        c0 = int((np.floor(lon.min()) - 1 + 180) / res) // block * block
        c1 = int((np.ceil(lon.max()) + 1 + 180) / res) // block * block
        r0 = int((90 - np.ceil(lat.max()) - 1) / res) // block * block
        r1 = int((90 - np.floor(lat.min()) + 1) / res) // block * block
        a = s.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
    m = np.isin(a, HUMAN_CODES).astype("float32")
    H, W = (m.shape[0] // block) * block, (m.shape[1] // block) * block
    sh = m[:H, :W].reshape(H // block, block, W // block, block).mean((1, 3))
    rr = ((90 - lat) / res - r0).astype(int) // block
    cc = ((lon + 180) / res - c0).astype(int) // block
    ok = (rr >= 0) & (rr < sh.shape[0]) & (cc >= 0) & (cc < sh.shape[1])
    out = np.full(lon.size, np.nan); out[ok] = sh[rr[ok], cc[ok]]
    return out


def main():
    import faulthandler; faulthandler.enable()
    c = CONFIG; out = gio.ensure_outdir(); t0 = time.time()
    spec = importlib.util.spec_from_file_location("m05", os.path.join(HERE, "05_eqi.py"))
    m05 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m05)
    arrs, years = m05.stack_years(out, c)
    if arrs is None:
        raise FileNotFoundError(f"[ATTR] {years}")
    core = sorted(set(years["efi"]) & set(years["esi"]) & set(years["epi"]))
    ypipe, _ = gio.eqi_annual_years(out)
    grid = gio.analysis_grid()
    mask = gio.read_mask()
    tr_core = np.load(os.path.join(out, "eqi_trend_core.npy"), mmap_mode="r")
    cand = mask & np.isfinite(np.asarray(tr_core)) & np.isfinite(np.load(os.path.join(out, "esi_trend.npy"), mmap_mode="r"))
    # stratified by ecosystem type (eco_type.tif from 08): PER_TYPE pixels per type so that shrub / wetland / sparse
    # have enough samples; weight = type pixel count / sampled count (used to re-weight pooled models)
    etype = np.nan_to_num(gio.read_tif(os.path.join(out, "eco_type.tif"))[0], nan=0).astype("uint8")
    rng = np.random.default_rng(SEED); picks, wts, tps = [], [], []
    for t in range(0, 6):
        ft = np.flatnonzero((cand & (etype == t)).ravel())
        if ft.size == 0:
            continue
        k = min(int(PER_TYPE * 1.15), ft.size)
        pk = np.sort(rng.choice(ft, size=k, replace=False))
        picks.append(pk); tps.append(np.full(k, t)); wts.append(np.full(k, ft.size / k))
    pick = np.concatenate(picks); tp = np.concatenate(tps); wt = np.concatenate(wts)
    flat = np.flatnonzero(cand.ravel())
    rows, cols = np.divmod(pick, mask.shape[1])
    print(f"[ATTR] core years {core[0]}-{core[-1]}; candidates {flat.size:,}; drawn {pick.size:,}", flush=True)

    D = {}
    ycore = np.array(core, float)
    D["EFI_change"] = ts(at_pts(arrs["efi"], [years["efi"].index(y) for y in core], rows, cols), ycore)
    D["ESI_change"] = ts(at_pts(arrs["esi"], [years["esi"].index(y) for y in core], rows, cols), ycore)
    D["EPI_change"] = ts(at_pts(arrs["epi"], [years["epi"].index(y) for y in core], rows, cols), ycore)
    eqi = np.load(os.path.join(out, "eqi_stack.npy"), mmap_mode="r")
    D["EQI_change"] = ts(at_pts(eqi, [list(ypipe).index(y) for y in core], rows, cols), ycore)
    print(f"[ATTR] pipeline slopes done {time.time() - t0:.0f}s", flush=True)

    lon, lat, x, y = lonlat(grid, rows, cols)
    base = os.path.join(CLIM)
    pre = sorted(glob.glob(os.path.join(base, "*降水量数据集", "Pre_Year", "pre_*.tif")))
    for nm, pat in (("AP_change", "*降水量数据集/Pre_Year/pre_{y}.tif"), ("PET_change", "*潜在蒸散发数据集/PET_Year/pet_{y}.tif"),
                    ("AI_change", "*干燥度数据集/ai_{y}.tif")):
        Y = []
        for yr in core:
            f = glob.glob(os.path.join(base, pat.format(y=yr)))
            if not f:
                raise FileNotFoundError(f"[ATTR] {nm}: no file for {yr} ({pat})")
            Y.append(sample_geo(f[0], lon, lat))
        D[nm] = ts(np.stack(Y), ycore)
        print(f"[ATTR] {nm} done {time.time() - t0:.0f}s", flush=True)
    res = {}
    for yr in core:        # sequential on purpose: threaded and process-parallel versions both exited silently here
        res.update([temp_year((yr, lon, lat))])
        print(f"[ATTR]   temperature {yr} {time.time() - t0:.0f}s", flush=True)
    D["AT_change"] = ts(np.stack([res[yr] for yr in core]), ycore)
    print(f"[ATTR] AT_change done {time.time() - t0:.0f}s", flush=True)

    hy = list(range(HFP_YEARS[0], HFP_YEARS[1] + 1))
    import rasterio
    H = []
    for yr in hy:
        with rasterio.open(os.path.join(c["HFP_DIR"], c["HFP_PATTERN"].format(year=yr))) as s:
            a = s.read(1); nod = s.nodata
        v = a[rows, cols].astype("float64"); v[v == nod] = np.nan; H.append(v)
    D["FP_change"] = ts(np.stack(H), np.array(hy, float))
    D["Lu_change"] = human_share(LU_YEARS[1], lon, lat) - human_share(LU_YEARS[0], lon, lat)
    print(f"[ATTR] FP / Lu done {time.time() - t0:.0f}s", flush=True)

    names = ["AT_change", "AP_change", "PET_change", "AI_change", "Lu_change", "FP_change",
             "EFI_change", "ESI_change", "EPI_change", "EQI_change"]
    M = np.column_stack([D[k] for k in names]); ok = np.isfinite(M).all(1)
    print(f"[ATTR] complete rows {int(ok.sum()):,} of {ok.size:,}; NaN per column: "
          + ", ".join(f"{k}={int((~np.isfinite(D[k])).sum())}" for k in names), flush=True)
    import pandas as pd
    df = pd.DataFrame(M[ok], columns=names); df["x"] = x[ok]; df["y"] = y[ok]; df["type"] = tp[ok]; df["w"] = wt[ok]
    parts = [g.sample(n=min(PER_TYPE, len(g)), random_state=SEED) for _, g in df.groupby("type")]
    df = pd.concat(parts).reset_index(drop=True)
    cnt = df.groupby("type").size()
    # weight = analysed pixels of the type / pixels kept for the type
    full = pd.Series({t: int((cand & (etype == t)).sum()) for t in cnt.index})
    df["w"] = df["type"].map(full / cnt)
    df[names + ["x", "y", "type", "w"]].to_csv(os.path.join(out, "attribution_table.csv"), index=False)
    # MGWR sample: pooled (weighted resample), plus forest (type 1) and grassland (type 3) subsets
    pool = df.sample(n=min(N_MGWR, len(df)), weights=df["w"], random_state=SEED + 1)
    def mg(sub):
        return pd.DataFrame({"x": sub["x"], "y": sub["y"], "dEQI": sub["EQI_change"], "precip": sub["AP_change"],
                             "temp": sub["AT_change"], "landuse": sub["Lu_change"], "footprint": sub["FP_change"]})
    mg(pool).to_csv(os.path.join(out, "mgwr_table.csv"), index=False)
    for t, nm in ((1, "forest"), (3, "grassland")):
        g = df[df["type"] == t]
        mg(g.sample(n=min(N_MGWR, len(g)), random_state=SEED + 2 + t)).to_csv(os.path.join(out, f"mgwr_table_{nm}.csv"), index=False)
    desc = df[names].describe().T[["mean", "std", "min", "max"]]
    with open(os.path.join(out, "attribution_sample_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"core years {core[0]}-{core[-1]}; HFP slope {HFP_YEARS}; Lu change {LU_YEARS[0]}->{LU_YEARS[1]} (3 km block); "
                f"N_rows={len(df)}\n{desc.to_string()}\n")
    print(desc.to_string(float_format=lambda v: f"{v:.5g}"))
    print(f"[ATTR] wrote attribution_table.csv ({len(df):,}); mgwr tables pooled/forest/grassland; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
