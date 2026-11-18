#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 12 — Independent validation
================================================================================
Paper 2.8 & 3.4. Three levels:
    (i)   field validation : EQI / EFI at field plots vs the plot's ecological score
    (ii)  product cross-validation : EFI vs an independent NPP / biomass product (optional rasters)
    (iii) trend agreement  : sign of the core-year EQI trend (05_eqi: eqi_trend_core.npy) vs an independent
          NDVI trend raster (optional)
Metrics: Pearson r, Pearson r^2, Spearman rho, RMSE and the 1:1 coefficient of determination R2_1to1
(1 - SSres/SStot against the line y = x; only meaningful when plot score and index share a scale), each with a
bootstrap 95% interval for the plot level.

There is NO synthetic fallback: without field_plots.csv the script stops with an error. Run it with --demo only to
test the code path (the outputs are then labelled DEMO and must never be quoted).

--------------------------------------------------------------------------------
INPUT
--------------------------------------------------------------------------------
Predictions (OUT_DIR, from 01/05):  eqi_equal.npy (whole-period EQI), efi_mean.npy, eqi_trend_core.npy
Field plots  field_plots.csv  (OUT_DIR, or the path in env EQI_PLOTS), one row per plot, columns
        field_score                    required, plot ecological score in [0, 1] (same scale as EQI)
        and ONE coordinate pair:
        lon, lat                       WGS84 degrees  (preferred; transformed to the analysis grid)
        x, y                           coordinates in the analysis-grid CRS (Albers, metres)
        row, col                       0-based indices into the analysis grid (H x W)
        optional: plot_id, site, ecosystem
    The index is read at the plot pixel; WINDOW=3 uses the mean of the valid pixels in the 3x3 window around it
    (500 m pixel vs a plot of tens of metres: single-pixel values are noisy). Plots outside the grid or on a pixel
    without an EQI (outside the stable natural mask) are dropped and counted, not silently ignored.
Optional independent rasters in OUT_DIR on the analysis grid, .tif or .npy: modis_npp, biomass, ndvi_trend.

OUTPUT (OUT_DIR or env EQI_VAL_OUT): validation_metrics.csv, validation_plots.csv, 12_validation_preview.png
Run:  python 12_validation.py    |    python 12_validation.py --demo
================================================================================
"""
from __future__ import annotations
import os, sys, csv, warnings
import numpy as np
import geo_io as gio
from geo_io import CONFIG

warnings.filterwarnings("ignore", category=RuntimeWarning)
RNG = np.random.default_rng(128)
WINDOW = 3                    # odd; 1 = single pixel
N_BOOT = 2000
N_SAMPLE_CELLS = 500_000      # pixels sampled for the product-level comparisons


# ----------------------------------------------------------------- metrics
def _pair(o, p):
    o, p = np.asarray(o, float), np.asarray(p, float)
    m = np.isfinite(o) & np.isfinite(p)
    return o[m], p[m]


def r2_1to1(obs, pred):
    o, p = _pair(obs, pred)
    if o.size < 2: return np.nan
    return float(1 - np.sum((o - p) ** 2) / (np.sum((o - o.mean()) ** 2) + 1e-12))


def rmse(obs, pred):
    o, p = _pair(obs, pred)
    return float(np.sqrt(np.mean((o - p) ** 2))) if o.size else np.nan


def pearson(obs, pred):
    o, p = _pair(obs, pred)
    if o.size < 3 or o.std() == 0 or p.std() == 0: return np.nan
    return float(np.corrcoef(o, p)[0, 1])


def spearman(obs, pred):
    from scipy.stats import spearmanr
    o, p = _pair(obs, pred)
    if o.size < 3: return np.nan
    return float(spearmanr(o, p).correlation)


def boot_ci(fn, obs, pred, n=N_BOOT):
    o, p = _pair(obs, pred)
    if o.size < 10: return (np.nan, np.nan)
    v = np.array([fn(o[i], p[i]) for i in (RNG.integers(0, o.size, o.size) for _ in range(n))])
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


def trend_agreement(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
    return (float(np.mean(np.sign(a[m]) == np.sign(b[m]))), int(m.sum())) if m.any() else (np.nan, 0)


# ----------------------------------------------------------------- plots -> grid
def plots_to_rc(df, grid):
    """Row / column of every plot on the analysis grid (float, may lie outside)."""
    cols = {c.lower(): c for c in df.columns}
    H, W = grid["height"], grid["width"]
    if "row" in cols and "col" in cols:
        return df[cols["row"]].to_numpy(float), df[cols["col"]].to_numpy(float), "row/col"
    if "lon" in cols and "lat" in cols:
        from rasterio.warp import transform as wtransform
        x, y = wtransform("EPSG:4326", grid["crs"], df[cols["lon"]].tolist(), df[cols["lat"]].tolist())
        src = "lon/lat (WGS84)"
    elif "x" in cols and "y" in cols:
        x, y, src = df[cols["x"]].to_numpy(float), df[cols["y"]].to_numpy(float), "x/y (grid CRS)"
    else:
        raise ValueError("[VAL] field_plots.csv needs lon+lat, x+y or row+col columns (plus field_score); "
                         f"found {list(df.columns)}")
    inv = ~grid["transform"]
    cc, rr = [], []
    for xi, yi in zip(x, y):
        c_, r_ = inv * (xi, yi); cc.append(c_); rr.append(r_)
    return np.floor(np.array(rr)), np.floor(np.array(cc)), src


def read_at(arr, rr, cc, win=WINDOW):
    """Value of arr at pixel (rr, cc); win > 1: mean of the finite pixels of the win x win window (NaN if none)."""
    H, W = arr.shape; h = win // 2; out = np.full(rr.size, np.nan); nvalid = np.zeros(rr.size, int)
    for i, (r, c) in enumerate(zip(rr, cc)):
        if not (np.isfinite(r) and np.isfinite(c)) or r < 0 or c < 0 or r >= H or c >= W:
            continue
        r, c = int(r), int(c)
        blk = np.asarray(arr[max(0, r - h):r + h + 1, max(0, c - h):c + h + 1], "float64")
        v = blk[np.isfinite(blk)]
        centre = arr[r, c]
        if np.isfinite(centre) and v.size:                  # the plot pixel itself must be analysed
            out[i] = v.mean(); nvalid[i] = v.size
    return out, nvalid


def load_raster(out, name):
    for ext in (".npy", ".tif"):
        p = os.path.join(out, name + ext)
        if os.path.exists(p):
            return np.load(p, mmap_mode="r") if ext == ".npy" else gio.read_tif(p)[0]
    return None


# ----------------------------------------------------------------- demo (only with --demo)
def generate_demo(out_dir):
    H = W = 60
    eqi = np.clip(0.55 + 0.12 * RNG.standard_normal((H, W)), 0, 1).astype("float32")
    efi = np.clip(eqi + 0.05 * RNG.standard_normal((H, W)), 0, 1).astype("float32")
    tr = (np.where(np.mgrid[0:H, 0:W][1] >= 2 * W // 3, -0.003, 0.005) + 0.001 * RNG.standard_normal((H, W))).astype("float32")
    rows = RNG.integers(0, H, 128); cols = RNG.integers(0, W, 128)
    import pandas as pd
    df = pd.DataFrame({"row": rows, "col": cols,
                       "field_score": np.clip(eqi[rows, cols] + 0.05 * RNG.standard_normal(128), 0, 1)})
    grid = {"height": H, "width": W, "crs": None, "transform": None}
    return dict(eqi=eqi, efi=efi, trend=tr, ndvi_trend=tr + 0.0008 * RNG.standard_normal((H, W)),
                modis=efi * 1000 + 50 * RNG.standard_normal((H, W)), biomass=None), df, grid


# ----------------------------------------------------------------- main
def main():
    demo = "--demo" in sys.argv
    out = gio.ensure_outdir(); vout = os.environ.get("EQI_VAL_OUT") or out
    os.makedirs(vout, exist_ok=True)
    if demo:
        print("[VAL] *** DEMO MODE: synthetic data, results are meaningless ***")
        R, df, grid = generate_demo(out)
        tag = "DEMO "
    else:
        need = ["eqi_equal.npy", "efi_mean.npy"]
        miss = [n for n in need if not os.path.exists(os.path.join(out, n))]
        fp = os.environ.get("EQI_PLOTS") or os.path.join(out, "field_plots.csv")
        if miss or not os.path.exists(fp):
            raise FileNotFoundError(
                f"[VAL] cannot validate: missing {miss + ([fp] if not os.path.exists(fp) else [])}. "
                "Provide field_plots.csv (columns: lon, lat, field_score; see the header) in OUT_DIR or set EQI_PLOTS. "
                "Run with --demo only to test the code.")
        import pandas as pd
        df = pd.read_csv(fp)
        if "field_score" not in {c.lower() for c in df.columns}:
            raise ValueError(f"[VAL] {fp} has no 'field_score' column ({list(df.columns)})")
        grid = gio.analysis_grid()
        R = dict(eqi=np.load(os.path.join(out, "eqi_equal.npy"), mmap_mode="r"),
                 efi=np.load(os.path.join(out, "efi_mean.npy"), mmap_mode="r"),
                 trend=load_raster(out, "eqi_trend_core"), ndvi_trend=load_raster(out, "ndvi_trend"),
                 modis=load_raster(out, "modis_npp"), biomass=load_raster(out, "biomass"))
        tag = ""
        H, W = R["eqi"].shape
        if (H, W) != (grid["height"], grid["width"]):
            raise ValueError(f"[VAL] eqi_equal.npy {(H, W)} differs from the analysis grid "
                             f"{(grid['height'], grid['width'])} (CONFIG ROI?)")
    fcol = {c.lower(): c for c in df.columns}["field_score"]
    field = df[fcol].to_numpy(float)
    rr, cc, src = plots_to_rc(df, grid)
    H, W = grid["height"], grid["width"]
    inside = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    eqi_p, n_eqi = read_at(R["eqi"], rr, cc); efi_p, _ = read_at(R["efi"], rr, cc)
    ok = np.isfinite(eqi_p) & np.isfinite(field)
    print(f"[VAL] {len(df)} plots, coordinates {src}: {int(inside.sum())} inside the grid, "
          f"{int(np.isfinite(eqi_p).sum())} on an analysed pixel; {len(df) - int(ok.sum())} dropped "
          f"({len(df) - int(inside.sum())} outside the grid, {int(inside.sum() - np.isfinite(eqi_p)[inside].sum())} "
          f"outside the stable natural mask or without EQI), window {WINDOW}x{WINDOW}")
    if ok.sum() < 10:
        raise RuntimeError(f"[VAL] only {int(ok.sum())} usable plots; check the coordinates / CRS")

    metrics = []
    def add(name, n, o, p, plot_level=True):
        r, rho = pearson(o, p), spearman(o, p)
        lo, hi = boot_ci(pearson, o, p) if plot_level else (np.nan, np.nan)
        metrics.append([tag + name, n, f"{r:.3f}", f"{lo:.3f}" if plot_level else "", f"{hi:.3f}" if plot_level else "",
                        f"{r * r:.3f}" if np.isfinite(r) else "nan", f"{rho:.3f}", f"{rmse(o, p):.3f}" if plot_level else "",
                        f"{r2_1to1(o, p):.3f}" if plot_level else ""])
    add("EQI vs field score", int(ok.sum()), field[ok], eqi_p[ok])
    okf = np.isfinite(efi_p) & np.isfinite(field)
    add("EFI vs field score", int(okf.sum()), field[okf], efi_p[okf])

    # product-level comparisons on a pixel sample
    mask = np.isfinite(np.asarray(R["efi"]))
    sidx = np.flatnonzero(mask.ravel())
    if sidx.size > N_SAMPLE_CELLS:
        sidx = np.sort(RNG.choice(sidx, N_SAMPLE_CELLS, replace=False))
    efi_s = np.asarray(R["efi"]).ravel()[sidx]
    for key, label in (("modis", "EFI vs independent NPP"), ("biomass", "EFI vs biomass")):
        if R.get(key) is not None:
            v = np.asarray(R[key]).ravel()[sidx]
            add(label, int(np.isfinite(v).sum()), v, efi_s, plot_level=False)
    if R.get("trend") is not None and R.get("ndvi_trend") is not None:
        a = np.asarray(R["trend"]).ravel()[sidx]; b = np.asarray(R["ndvi_trend"]).ravel()[sidx]
        agree, na = trend_agreement(a, b)
        metrics.append([tag + "EQI vs NDVI trend, sign agreement", na, f"{agree * 100:.1f}%", "", "", "", "", "", ""])
    elif not demo:
        print("[VAL] no independent product rasters (modis_npp / biomass / ndvi_trend) in OUT_DIR: "
              "product-level and trend-agreement checks skipped")

    with open(os.path.join(vout, "validation_metrics.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "n", "pearson_r_or_agreement", "r_ci_low", "r_ci_high", "pearson_r2", "spearman_rho",
                    "RMSE", "R2_1to1"])
        w.writerows(metrics)
    pd_out = df.copy(); pd_out["eqi_at_plot"] = eqi_p; pd_out["efi_at_plot"] = efi_p
    pd_out["used"] = ok.astype(int); pd_out["n_valid_pixels_in_window"] = n_eqi
    pd_out.to_csv(os.path.join(vout, "validation_plots.csv"), index=False, encoding="utf-8-sig")
    print("[VAL] metrics:")
    for m in metrics:
        print("        " + f"{m[0]:44s} n={m[1]:>8}  r/agr {m[2]:>7}  [{m[3]}, {m[4]}]  rho {m[6]}  RMSE {m[7]}  R2_1to1 {m[8]}")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(4.5, 4.2))
        plt.scatter(field[ok], eqi_p[ok], s=14, alpha=.6, color="#2e7d32")
        lim = [0, 1]; plt.plot(lim, lim, "k--", lw=.8)
        plt.xlabel("Field ecological score"); plt.ylabel("EQI at plot")
        plt.title(f"{tag}Validation (r={pearson(field[ok], eqi_p[ok]):.2f}, n={int(ok.sum())})")
        plt.xlim(lim); plt.ylim(lim); plt.tight_layout()
        plt.savefig(os.path.join(vout, "12_validation_preview.png"), dpi=120)
        print(f"[VAL] preview -> {vout}/12_validation_preview.png")
    except Exception as e:
        print(f"[VAL] (preview skipped: {e})")
    print("[VAL] done.")


if __name__ == "__main__":
    main()
