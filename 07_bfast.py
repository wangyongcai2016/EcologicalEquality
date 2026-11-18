#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 07 — Trend-breakpoint detection (PELT on the differenced series)   [no-args / VSCode-run]
================================================================================
Paper 2.3: detect trend breakpoints in each pixel's annual EQI series (timing,
pre/post slopes). A slope change becomes a mean shift in the first difference, so
PELT (l2 cost, min segment 3, jump 1) is run on the differenced series. This is a
change-point analysis, NOT BFAST (no season-trend decomposition).

Conventions
  * a change point b in the differenced series d (d[i] = s[i+1] - s[i]) separates d[:b] and d[b:], i.e. the slope
    changes at the VERTEX s[b]; the break year is years[b] (no +1; checked on synthetic V-shaped series).
  * penalty = c * sigma^2 * ln(n), n = len(d), sigma = MAD-based sd of the pixel's d (floored at SIGMA_FLOOR),
    c = CONFIG["BREAK_PENALTY_C"]. A fixed penalty is not used: it either flags noise everywhere or nothing.
  * only pixels with a complete series in the selected years are analysed (ruptures needs a contiguous series).
  * dominant break = the one with the largest |slope_post - slope_pre|.
  * with min segment 3 the earliest/latest detectable vertex is years[3] / years[-4].

Years: from 05's eqi_yearly_mean.csv; CONFIG["TREND_YEARS"] "core" (default, normally 2002-2020) or "all".

HOW TO RUN: run 05_eqi first (writes eqi_stack.npy), then this file.  pip install ruptures==1.1.9
INPUT (CONFIG['OUT_DIR']): eqi_stack.npy, eqi_yearly_mean.csv
OUTPUT: bfast_nbreaks/break_time/slope_pre/slope_post .npy (+ .tif), bfast_break_years.csv, 07_bfast_preview.png
================================================================================
"""
from __future__ import annotations
import os, csv, time, warnings
import numpy as np
import geo_io as gio
from geo_io import CONFIG

warnings.filterwarnings("ignore", category=RuntimeWarning)
RNG = np.random.default_rng(13)
MIN_SIZE = 3
SIGMA_FLOOR = 0.01          # lower bound of the differenced-series sd (a nearly linear series must not get ~0 penalty)
ROWS = 32

try:
    import ruptures as rpt
    _HAS_RPT = True
except Exception:
    _HAS_RPT = False


def theil_sen(y, x=None):
    y = np.asarray(y, float); x = np.arange(y.size, dtype=float) if x is None else np.asarray(x, float)
    m = np.isfinite(y); y, x = y[m], x[m]; n = y.size
    if n < 3: return np.nan
    sl = [(y[i+1:]-y[i])/(x[i+1:]-x[i]) for i in range(n-1)]
    return float(np.median(np.concatenate(sl)))


def penalty_for(d, c):
    """c * sigma_d^2 * ln(n). sigma_d is the noise sd of the differenced series d, estimated from the SECOND
    difference dd = diff(d) (variance 6*s^2 for white noise s, sigma_d^2 = 2*s^2, so sigma_d = MAD-sd(dd)/sqrt(3)).
    dd is zero for a piecewise-linear signal except at the vertex, so a real break does not inflate the estimate
    (the MAD of d itself does: a +-0.02 slope flip made a V-shaped series undetectable)."""
    dd = np.diff(d)
    sig = 1.4826 * np.median(np.abs(dd - np.median(dd))) / np.sqrt(3.0)
    return c * max(sig, SIGMA_FLOOR) ** 2 * np.log(d.size)


def detect_breaks(series, c, min_size=MIN_SIZE):
    """Vertex indices (into `series`) of the slope breaks."""
    s = np.asarray(series, float)
    if s.size < 2 * min_size + 1:
        return []
    d = np.diff(s)
    bk = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(d).predict(pen=penalty_for(d, c))
    return [b for b in bk if b < d.size]


def dominant_break(series, bkps):
    """(vertex index or -1, slope before, slope after). Slopes per time step; the two fits share the vertex."""
    s = np.asarray(series, float)
    if not bkps:
        sl = theil_sen(s); return -1, sl, sl
    best, best_mag, best_pair = bkps[0], -1.0, (np.nan, np.nan)
    for b in bkps:
        pre, post = theil_sen(s[:b + 1]), theil_sen(s[b:])
        mag = abs(post - pre)
        if np.isfinite(mag) and mag > best_mag:
            best, best_mag, best_pair = b, mag, (pre, post)
    return best, best_pair[0], best_pair[1]


def _block(path, sel, r0, r1, c):
    st = np.load(path, mmap_mode="r")
    Y = np.asarray(st[sel, r0:r1], "float64")
    T, h, w = Y.shape
    Yc = Y.reshape(T, h * w)
    full = np.isfinite(Yc).all(0)
    nb = np.full(h * w, np.nan, "float32"); bi = np.full(h * w, np.nan, "float32")
    sp = np.full(h * w, np.nan, "float32"); spo = np.full(h * w, np.nan, "float32")
    for j in np.flatnonzero(full):
        s = Yc[:, j]
        if np.ptp(s) < 1e-9:
            nb[j] = 0; sp[j] = spo[j] = 0.0; continue
        bk = detect_breaks(s, c)
        idx, pre, post = dominant_break(s, bk)
        nb[j] = len(bk); bi[j] = idx if idx >= 0 else np.nan; sp[j] = pre; spo[j] = post
    return r0, dict(nbreaks=nb.reshape(h, w), break_idx=bi.reshape(h, w),
                    slope_pre=sp.reshape(h, w), slope_post=spo.reshape(h, w), complete=full.reshape(h, w))


def run(path, sel, c, rows=ROWS, n_jobs=None):
    st = np.load(path, mmap_mode="r"); H, W = st.shape[1:]
    keys = ("nbreaks", "break_idx", "slope_pre", "slope_post")
    res = {k: np.full((H, W), np.nan, "float32") for k in keys}
    complete = np.zeros((H, W), bool)
    starts = list(range(0, H, rows)); nj = min(n_jobs or CONFIG["N_JOBS"], len(starts))
    t0 = time.time()
    if nj > 1:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=nj, verbose=0)(delayed(_block)(path, sel, r0, min(H, r0 + rows), c) for r0 in starts)
    else:
        parts = [_block(path, sel, r0, min(H, r0 + rows), c) for r0 in starts]
    for r0, d in parts:
        n = d["nbreaks"].shape[0]
        for k in keys:
            res[k][r0:r0 + n] = d[k]
        complete[r0:r0 + n] = d["complete"]
    print(f"[BREAK] {len(starts)} blocks, {nj} workers, {time.time() - t0:.0f}s")
    return res, complete


def generate_demo():
    n, H, W, tb = 19, 30, 30, 9
    yy, xx = np.mgrid[0:H, 0:W]
    plateau = (xx >= 2*W//3) & (yy >= H//2); t = np.arange(n)
    nonp = 0.50 + 0.006*t
    plat = np.where(t < tb, 0.50+0.006*t, 0.50+0.006*tb-0.014*(t-tb))
    stack = np.empty((n, H, W), "float32")
    for k in range(n):
        stack[k] = np.clip(np.where(plateau, plat[k], nonp[k]) + 0.006*RNG.standard_normal((H, W)), 0, 1)
    return stack


def main():
    c = CONFIG; out = gio.ensure_outdir()
    if not _HAS_RPT:
        raise ImportError("[BREAK] 'ruptures' is required: pip install ruptures==1.1.9 "
                          "(without it the old script silently returned no breaks)")
    cpen = float(c.get("BREAK_PENALTY_C", 2.0))
    path = os.path.join(out, "eqi_stack.npy")
    if os.path.exists(path):
        years, core = gio.eqi_annual_years(out)
    elif c["ALLOW_DEMO"]:
        print("[BREAK] eqi_stack.npy not found (run 05_eqi.py first); running synthetic demo")
        path = os.path.join(out, "_demo_eqi_stack.npy"); np.save(path, generate_demo())
        years = np.arange(2002, 2021); core = np.ones(19, bool)
    else:
        raise FileNotFoundError("[BREAK] need eqi_stack.npy from 05_eqi.py")
    mode = str(c.get("TREND_YEARS", "core")).lower()
    sel = np.flatnonzero(core) if mode == "core" else np.arange(len(years))
    ysel = years[sel]
    print(f"[BREAK] {path}: using TREND_YEARS='{mode}': {ysel[0]}-{ysel[-1]} ({len(sel)} yr); "
          f"PELT l2, min segment {MIN_SIZE}, penalty = {cpen:g}*sigma^2*ln(n); detectable vertex years "
          f"{ysel[MIN_SIZE]}-{ysel[-1 - MIN_SIZE]}")

    prof = None
    tif = os.path.join(out, "efi_mean.tif")
    if gio._HAS_RIO and os.path.exists(tif):
        import rasterio
        with rasterio.open(tif) as s:
            prof = s.profile

    res, complete = run(path, sel, cpen)
    bt = np.where(np.isfinite(res["break_idx"]), ysel[np.nan_to_num(res["break_idx"]).astype(int)], np.nan).astype("float32")
    res["break_time"] = bt
    for k, fn in [("nbreaks","bfast_nbreaks"),("break_time","bfast_break_time"),
                  ("slope_pre","bfast_slope_pre"),("slope_post","bfast_slope_post")]:
        np.save(os.path.join(out, fn+".npy"), res[k].astype(np.float32))
        if prof is not None:
            gio.write_tif(os.path.join(out, fn+".tif"), res[k], prof)

    nvalid = int(np.isfinite(res["nbreaks"]).sum()); nhave = int((res["nbreaks"] > 0).sum())
    anyv = np.isfinite(np.load(path, mmap_mode="r")[sel[0]]).sum()
    print(f"[BREAK] analysed {nvalid:,} pixels with a complete series ({100*nvalid/max(int(anyv),1):.1f}% of pixels valid in "
          f"{ysel[0]}); pixels with gaps are skipped")
    if nhave:
        b = res["break_time"][res["nbreaks"] > 0]
        steep = (res["slope_post"][res["nbreaks"] > 0] < res["slope_pre"][res["nbreaks"] > 0]).mean() * 100
        print(f"[BREAK] pixels with >=1 break: {100*nhave/nvalid:.1f}%  (mean breaks/pixel {np.nanmean(res['nbreaks']):.3f})")
        print(f"[BREAK] dominant-break median year {np.median(b):.0f} [{np.percentile(b,25):.0f}, {np.percentile(b,75):.0f}]; "
              f"{steep:.0f}% have a lower slope after the break")
        up = ((res["slope_pre"] <= 0) & (res["slope_post"] > 0) & (res["nbreaks"] > 0)).sum()
        dn = ((res["slope_pre"] >= 0) & (res["slope_post"] < 0) & (res["nbreaks"] > 0)).sum()
        print(f"[BREAK] sign reversal at the break: decline->increase {100*up/nhave:.1f}%, increase->decline {100*dn/nhave:.1f}% of breaking pixels")
        yrs_u, cnt = np.unique(b.astype(int), return_counts=True)
        with open(os.path.join(out, "bfast_break_years.csv"), "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(["break_year", "n_pixels", "share_of_breaking_pixels"])
            wr.writerows([[int(y), int(n), f"{n/nhave:.4f}"] for y, n in zip(yrs_u, cnt)])
        edge = cnt[[0, -1]].sum() / nhave * 100
        print(f"[BREAK] share of breaks in the first/last detectable year ({yrs_u[0]}, {yrs_u[-1]}): {edge:.0f}% "
              f"(an edge pile-up points to end-of-record effects, not real synchronous change)")
    else:
        print("[BREAK] no breakpoints detected")
    if prof is not None:
        print("[BREAK] wrote GeoTIFFs: bfast_*.tif")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        k = max(1, max(bt.shape) // 800)
        fig, ax = plt.subplots(1, 2, figsize=(8, 3.4))
        im0=ax[0].imshow(bt[::k, ::k], cmap="plasma"); ax[0].set_title("Dominant break year"); plt.colorbar(im0,ax=ax[0],shrink=.8)
        im1=ax[1].imshow((res["slope_post"]-res["slope_pre"])[::k, ::k], cmap="RdBu", vmin=-0.03, vmax=0.03); ax[1].set_title("Post - Pre slope"); plt.colorbar(im1,ax=ax[1],shrink=.8)
        plt.tight_layout(); plt.savefig(os.path.join(out, "07_bfast_preview.png"), dpi=120)
        print(f"[BREAK] preview -> {out}/07_bfast_preview.png")
    except Exception as e:
        print(f"[BREAK] (preview skipped: {e})")
    if os.path.basename(path).startswith("_demo"):
        os.remove(path)
    print("[BREAK] done.")


if __name__ == "__main__":
    main()
