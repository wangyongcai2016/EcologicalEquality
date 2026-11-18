#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 03 — SPEI-12 drought index & drought-pressure field   [aligned monthly rasters | tiled, parallel]
================================================================================
Paper section 2.5 (EPI), Eq. 10:   D = 1 - exp(-F * I)

    water balance  D_t = P_t - PET_t   ->  12-month accumulation  ->  log-logistic fit  ->  z-score (SPEI-12)
    drought month  SPEI < -1;  F = share of drought months,  I = mean |SPEI| over drought months.

Method notes (differences from the previous version, which looped over pixels with scipy.stats.fisk.fit):
  * The 3-parameter log-logistic is fitted per CALENDAR MONTH (12 fits per pixel), as in the standard SPEI
    (Vicente-Serrano et al. 2010), by L-moments from unbiased probability-weighted moments (Hosking 1990) --
    fully vectorised over pixels. If the fit is invalid for a pixel-month (l2 <= 0 or |k| >= 0.99) the empirical
    rank probability rank/(N+1) is used instead; the share of such cases is printed. Checked on synthetic samples:
    parameters recovered to 1 %, SPEI mean 0 / sd 0.97 / 17 % below -1 for N = 20-21 whatever the location of the
    data (the previous biased-PWM version gave 36 % below -1 on real arid pixels).
  * Period = START_YEAR .. min(END_YEAR, LU_END_YEAR) so that it matches the land-cover years used by EPI
    (the paper analyses 2000-2020). Only natural, assessment-unit pixels are computed (geo_io.read_mask).

HOW TO RUN
  1. python 00_align_monthly.py   (once; must have produced pre AND pet in PRECIP_ALIGN_DIR / PET_ALIGN_DIR)
  2. python 03_spei.py
  Pilot: set CONFIG["ROI"] (or env EQI_ROI) first, as for 02_esi.py.

INPUT  : aligned monthly precipitation + PET (geo_io: PRECIP_ALIGN_DIR, PET_ALIGN_DIR), UNITS_TIF, STABLE_TIF
OUTPUT (CONFIG['OUT_DIR']): drought_pressure.tif/.npy (+ drought.npy alias read by 11), spei_last.tif/.npy,
                            03_spei_preview.png
================================================================================
"""
from __future__ import annotations
import os, time, warnings
import numpy as np
from scipy.special import ndtri
import geo_io as gio
from geo_io import CONFIG

warnings.filterwarnings("ignore", category=RuntimeWarning)
RNG = np.random.default_rng(11)
SCALE = 12
DROUGHT_THRESHOLD = -1.0
CHUNK = 8192           # pixels per vectorised block (bounds the float64 working set)
MIN_YEARS = 10         # minimum number of accumulated values per calendar month for a fit


def study_years():
    start = CONFIG["START_YEAR"]
    end = min(CONFIG["END_YEAR"], CONFIG.get("LU_END_YEAR", CONFIG["END_YEAR"]))
    return start, end


# ------------------------------------------------------------ log-logistic fit (vectorised, columns = pixels)
def _loglogistic_cdf(x):
    """x : (N, k) accumulated water balance of ONE calendar month. Returns (cdf (N,k), fallback mask (k,)).

    Three-parameter log-logistic = generalised logistic (GLO) with location xi, scale a, shape k, fitted by
    L-moments from UNBIASED probability-weighted moments (Hosking 1990; Vicente-Serrano et al. 2010 use the same
    estimator). The biased PWM with plotting positions (i-0.35)/N that an earlier version used is only valid for
    N in the hundreds: with N = 21 years per calendar month it returned beta < 1 or a lower bound above the data
    for most pixels, i.e. a rank fallback or SPEI at the floor, and a share of SPEI < -1 of 36 % instead of 16 %.
    """
    N = x.shape[0]
    order = np.argsort(x, axis=0)
    xs = np.take_along_axis(x, order, axis=0)
    i = np.arange(1, N + 1, dtype="float64")[:, None]
    b0 = xs.mean(0)
    b1 = ((N - i) / (N - 1.0) * xs).mean(0)
    b2 = ((N - i) * (N - i - 1.0) / ((N - 1.0) * (N - 2.0)) * xs).mean(0)
    l1, l2, l3 = b0, b0 - 2 * b1, b0 - 6 * b1 + 6 * b2        # L-moments
    with np.errstate(invalid="ignore", divide="ignore"):
        k = -l3 / l2                                           # GLO shape (tau3 = -k)
    good = np.isfinite(k) & (l2 > 0) & (np.abs(k) < 0.99)
    kk = np.where(good, k, 0.1)
    kk = np.where(np.abs(kk) < 1e-5, 1e-5, kk)
    a = l2 * np.sin(np.pi * kk) / (np.pi * kk)
    xi = l1 - a * (1.0 / kk - np.pi / np.sin(np.pi * kk))
    arg = 1.0 - kk * (x - xi) / a
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        Y = -np.log(np.where(arg > 0, arg, np.nan)) / kk
        cdf = 1.0 / (1.0 + np.exp(-Y))
    cdf = np.where(arg > 0, cdf, np.where(kk > 0, 1.0, 0.0)[None, :])   # outside the support of the fitted law
    rank = np.empty(order.shape, "float64")
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(1, N + 1, dtype="float64")[:, None], order.shape), axis=0)
    cdf = np.where(good[None, :], cdf, rank / (N + 1.0))        # invalid fit -> empirical rank probability
    return np.clip(cdf, 1e-6, 1 - 1e-6), ~good


def spei_cols(precip, pet, scale=SCALE):
    """precip, pet : (T, n) monthly series (Jan first). Returns SPEI (T, n) and the fallback share."""
    P = np.asarray(precip, np.float64)
    E = np.asarray(pet, np.float64)
    T, n = P.shape
    out = np.full((T, n), np.nan)
    D = P - E
    ok = np.isfinite(D).all(0)
    if not ok.any() or T < scale + 12 * MIN_YEARS:
        return out, 0.0
    cs = np.cumsum(np.where(ok, D, 0.0), axis=0)
    acc = cs.copy()
    acc[scale:] = cs[scale:] - cs[:-scale]
    acc[: scale - 1] = np.nan
    n_fb = n_tot = 0
    cols = np.flatnonzero(ok)
    for m in range(12):
        idx = np.arange(m, T, 12)
        idx = idx[idx >= scale - 1]
        if idx.size < MIN_YEARS:
            continue
        x = acc[idx][:, cols]
        cdf, fb = _loglogistic_cdf(x)
        out[np.ix_(idx, cols)] = ndtri(cdf)
        n_fb += int(fb.sum()); n_tot += fb.size
    return out, (n_fb / max(n_tot, 1))


def drought_pressure_cols(spei, threshold=DROUGHT_THRESHOLD):
    """spei : (T, n) -> drought pressure (n,), latest SPEI (n,)."""
    fin = np.isfinite(spei)
    nv = fin.sum(0)
    dm = fin & (spei < threshold)
    freq = dm.sum(0) / np.maximum(nv, 1)
    inten = np.where(dm, np.abs(spei), 0.0).sum(0) / np.maximum(dm.sum(0), 1)
    dp = 1.0 - np.exp(-freq * inten)
    dp[nv == 0] = np.nan
    last = np.full(spei.shape[1], np.nan)
    has = nv > 0
    last[has] = spei[::-1][np.argmax(fin[::-1], axis=0), np.arange(spei.shape[1])][has]
    return dp, last


def compute_cols(precip, pet, scale=SCALE):
    """Chunked wrapper: (T, n) -> dp (n,), last (n,), fallback share."""
    n = precip.shape[1]
    dp = np.full(n, np.nan, "float32")
    last = np.full(n, np.nan, "float32")
    fb_sum = 0.0
    nb = 0
    for s in range(0, n, CHUNK):
        sl = slice(s, min(n, s + CHUNK))
        sp, fb = spei_cols(precip[:, sl], pet[:, sl], scale)
        d, l = drought_pressure_cols(sp)
        dp[sl] = d
        last[sl] = l
        fb_sum += fb
        nb += 1
    return dp, last, fb_sum / max(nb, 1)


# ------------------------------------------------------------ tiled real run
def process_tile(win_t, start, end):
    from rasterio.windows import Window
    r0, c0, h, w = win_t
    win = Window(c0, r0, w, h)
    mask = gio.read_mask(win)
    if not mask.any():
        return r0, c0, h, w, None, None, None, 0.0
    midx = np.flatnonzero(mask.ravel())
    pre, _ = gio.read_monthly_window("pre", start, end, win)
    pet, _ = gio.read_monthly_window("pet", start, end, win)
    pre = pre.reshape(pre.shape[0], -1)[:, midx]
    pet = pet.reshape(pet.shape[0], -1)[:, midx]
    dp, last, fb = compute_cols(pre, pet)
    return r0, c0, h, w, midx, dp, last, fb


def _aligned_ready():
    return all(gio.files_available(gio.align_dir(k)) for k in ("PRECIP_ALIGN_DIR", "PET_ALIGN_DIR"))


def run_real(out):
    from joblib import Parallel, delayed
    c = CONFIG
    start, end = study_years()
    grid = gio.analysis_grid()
    H, W = grid["height"], grid["width"]
    t = c["TILE"]
    tiles = [(r, cc, min(t, H - r), min(t, W - cc)) for r in range(0, H, t) for cc in range(0, W, t)]
    print(f"[SPEI] grid {H}x{W}{' (ROI)' if c.get('ROI') else ''}, {start}-{end}, scale {SCALE}, "
          f"{len(tiles)} tiles, workers={c['N_JOBS']}")
    dp = np.full((H, W), np.nan, "float32")
    last = np.full((H, W), np.nan, "float32")
    t0 = time.time()
    npx = 0
    fbs = []
    gen = Parallel(n_jobs=c["N_JOBS"], return_as="generator_unordered")(
        delayed(process_tile)(tl, start, end) for tl in tiles)
    for i, (r0, c0, h, w, midx, d, l, fb) in enumerate(gen, 1):
        if midx is not None:
            full = np.full(h * w, np.nan, "float32"); full[midx] = d; dp[r0:r0 + h, c0:c0 + w] = full.reshape(h, w)
            full = np.full(h * w, np.nan, "float32"); full[midx] = l; last[r0:r0 + h, c0:c0 + w] = full.reshape(h, w)
            npx += midx.size
            fbs.append(fb)
        if i % 10 == 0 or i == len(tiles):
            el = time.time() - t0
            print(f"[SPEI] tiles {i}/{len(tiles)}  pixels {npx:,}  {el:.0f}s  "
                  f"ETA {el / i * (len(tiles) - i):.0f}s", flush=True)
    if fbs:
        print(f"[SPEI] rank-probability fallback in {100 * float(np.mean(fbs)):.2f}% of pixel-month fits")
    return dp, last, grid


def generate_demo():
    c = CONFIG
    H = W = 30
    T = (min(c["END_YEAR"], c.get("LU_END_YEAR", c["END_YEAR"])) - c["START_YEAR"] + 1) * 12
    yy, xx = np.mgrid[0:H, 0:W]
    plateau = ((xx >= 2 * W // 3) & (yy >= H // 2)).ravel()
    season = 1 + 0.5 * np.sin(2 * np.pi * (np.arange(T) % 12) / 12)
    n = H * W
    yr = (np.arange(T) // 12)[:, None]
    pre = np.clip(50 * season[:, None] + 10 * RNG.standard_normal((T, n)) - 0.6 * yr * plateau[None, :], 0, None)
    pet = np.clip(45 * season[:, None] + 6 * RNG.standard_normal((T, n)), 0, None)
    dp, last, fb = compute_cols(pre, pet)
    return dp.reshape(H, W), last.reshape(H, W), None


def main():
    c = CONFIG
    out = gio.ensure_outdir()
    real_cfg = gio.files_available(c["UNITS_TIF"]) and gio.files_available(c["PRECIP_DIR"])
    if real_cfg and not _aligned_ready():
        raise FileNotFoundError(
            "[SPEI] aligned precipitation / PET not found in "
            f"{gio.align_dir('PRECIP_ALIGN_DIR')} / {gio.align_dir('PET_ALIGN_DIR')}.\n"
            "      Run 00_align_monthly.py (keys npp, pre, pet) first.")
    if real_cfg:
        dp, last, grid = run_real(out)
    elif c["ALLOW_DEMO"]:
        print("[SPEI] data paths not found; running synthetic demo")
        dp, last, grid = generate_demo()
    else:
        raise FileNotFoundError("[SPEI] configured data not found and ALLOW_DEMO=False")

    np.save(os.path.join(out, "drought_pressure.npy"), dp)
    np.save(os.path.join(out, "drought.npy"), dp)                      # alias read by 11_monte_carlo
    np.save(os.path.join(out, "spei_last.npy"), last)
    if grid is not None:
        gio.write_tif(os.path.join(out, "drought_pressure.tif"), dp, grid)
        gio.write_tif(os.path.join(out, "spei_last.tif"), last, grid)
        print("[SPEI] wrote GeoTIFFs: drought_pressure.tif, spei_last.tif")
    v = dp[np.isfinite(dp)]
    if v.size:
        print(f"[SPEI] drought-pressure mean = {v.mean():.3f}  [{v.min():.3f}, {v.max():.3f}]  "
              f"valid pixels {v.size:,}")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        sd = max(1, dp.shape[0] // 900)
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
        im0 = ax[0].imshow(last[::sd, ::sd], cmap="BrBG", vmin=-2.5, vmax=2.5); ax[0].set_title("SPEI-12 (latest)")
        plt.colorbar(im0, ax=ax[0], shrink=.8)
        im1 = ax[1].imshow(dp[::sd, ::sd], cmap="OrRd", vmin=0, vmax=1); ax[1].set_title("Drought pressure")
        plt.colorbar(im1, ax=ax[1], shrink=.8)
        for a in ax: a.axis("off")
        plt.tight_layout(); plt.savefig(os.path.join(out, "03_spei_preview.png"), dpi=120)
        print(f"[SPEI] preview -> {os.path.join(out, '03_spei_preview.png')}")
    except Exception as e:
        print(f"[SPEI] (preview skipped: {e})")
    print("[SPEI] done.")


if __name__ == "__main__":
    main()
